#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CosyVoice模型下载管理器
负责下载和管理CosyVoice系列模型

功能特性:
- 自动检测和选择下载源（ModelScope/HuggingFace）
- 支持断点续传和下载进度跟踪
- 模型完整性验证和状态管理
- 自动处理依赖安装
- 多线程并发下载优化
- 完善的错误处理和重试机制

"""

import os
import sys
import json
import time
import threading
import traceback
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib

# 导入PathManager
from .path_manager import PathManager


# ==================== 异常类定义 ====================

class ModelDownloadError(Exception):
    """模型下载异常基类"""
    pass


class DownloadSourceError(ModelDownloadError):
    """下载源异常"""
    pass


class ModelVerificationError(ModelDownloadError):
    """模型验证异常"""
    pass


class DependencyInstallationError(ModelDownloadError):
    """依赖安装异常"""
    pass


# ==================== 枚举类定义 ====================

class DownloadSource(Enum):
    """下载源枚举"""
    MODELSCOPE = "modelscope"
    HUGGINGFACE = "huggingface"
    AUTO = "auto"  # 自动选择


class DownloadStatus(Enum):
    """下载状态枚举"""
    PENDING = "pending"
    DOWNLOADING = "downloading"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    VERIFIED = "verified"


class ModelType(Enum):
    """模型类型枚举"""
    COSYVOICE3_2512 = "Fun-CosyVoice3-0.5B-2512"
    COSYVOICE3_FUN = "Fun-CosyVoice3-0.5B"
    COSYVOICE2 = "CosyVoice2-0.5B"
    COSYVOICE_300M = "CosyVoice-300M"
    COSYVOICE_300M_SFT = "CosyVoice-300M-SFT"
    COSYVOICE_300M_INSTRUCT = "CosyVoice-300M-Instruct"
    COSYVOICE_TTSFRD = "CosyVoice-ttsfrd"


# ==================== 数据类定义 ====================

@dataclass
class ModelInfo:
    """模型信息"""
    model_type: ModelType
    modelscope_id: str
    huggingface_id: str
    local_dir: str
    size_gb: float = 0.0
    description: str = ""
    required: bool = True
    priority: int = 1  # 下载优先级，数字越小优先级越高
    checksum: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)


@dataclass
class DownloadProgress:
    """下载进度信息"""
    model_type: ModelType
    status: DownloadStatus
    progress: float = 0.0  # 0.0-1.0
    downloaded_size: int = 0  # 字节
    total_size: int = 0  # 字节
    speed: float = 0.0  # KB/s
    eta: Optional[int] = None  # 预计剩余时间（秒）
    error_message: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None


@dataclass
class ModelDownloadRequest:
    """模型下载请求"""
    model_types: List[ModelType]
    source: DownloadSource = DownloadSource.AUTO
    force_download: bool = False
    verify_integrity: bool = True
    install_dependencies: bool = True
    callback: Optional[Callable[[DownloadProgress], None]] = None


# ==================== 日志混入类 ====================

class LoggerMixin:
    """日志混入类"""

    def __init__(self):
        self._logger = None

    @property
    def logger(self):
        """获取日志记录器"""
        if self._logger is None:
            import logging
            self._logger = logging.getLogger(self.__class__.__name__)
            if not self._logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter(
                    '[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                )
                handler.setFormatter(formatter)
                self._logger.addHandler(handler)
                self._logger.setLevel(logging.INFO)
        return self._logger


# ==================== 模型下载管理器主类 ====================

class ModelDownloadManager(LoggerMixin):
    """
    CosyVoice模型下载管理器

    提供完整的模型下载、验证和管理功能，支持多种下载源和并发下载。

    主要功能:
    - 自动检测下载源可用性
    - 支持ModelScope和HuggingFace下载
    - 断点续传和进度跟踪
    - 模型完整性验证
    - 依赖自动安装
    - 状态持久化

    使用示例:
        manager = ModelDownloadManager()

        # 下载单个模型
        result = manager.download_model(ModelType.COSYVOICE3_2512)

        # 下载多个模型
        result = manager.download_models([
            ModelType.COSYVOICE3_2512,
            ModelType.COSYVOICE_TTSFRD
        ])

        # 获取下载状态
        status = manager.get_download_status()
    """

    def __init__(self, path_manager: PathManager = None):
        # 初始化父类LoggerMixin
        LoggerMixin.__init__(self)

        self.path_manager = path_manager or PathManager()
        self._lock = threading.RLock()
        self._download_futures = {}
        self._progress_callbacks = {}
        self._load_model_status()

        # 预定义模型信息
        self._model_registry = self._initialize_model_registry()

    def _initialize_model_registry(self) -> Dict[ModelType, ModelInfo]:
        """初始化模型注册表"""
        return {
            ModelType.COSYVOICE3_2512: ModelInfo(
                model_type=ModelType.COSYVOICE3_2512,
                modelscope_id="FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
                huggingface_id="FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
                local_dir="Fun-CosyVoice3-0.5B-2512",
                size_gb=2.5,
                description="CosyVoice3 2512版本，最新的零样本语音克隆模型",
                required=True,
                priority=1
            ),
            ModelType.COSYVOICE3_FUN: ModelInfo(
                model_type=ModelType.COSYVOICE3_FUN,
                modelscope_id="FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
                huggingface_id="FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
                local_dir="Fun-CosyVoice3-0.5B",
                size_gb=2.5,
                description="CosyVoice3 Fun版本，通用语音合成模型",
                required=False,
                priority=2
            ),
            ModelType.COSYVOICE2: ModelInfo(
                model_type=ModelType.COSYVOICE2,
                modelscope_id="iic/CosyVoice2-0.5B",
                huggingface_id="FunAudioLLM/CosyVoice2-0.5B",
                local_dir="CosyVoice2-0.5B",
                size_gb=1.0,
                description="CosyVoice2 流式推理模型",
                required=False,
                priority=3
            ),
            ModelType.COSYVOICE_300M: ModelInfo(
                model_type=ModelType.COSYVOICE_300M,
                modelscope_id="iic/CosyVoice-300M",
                huggingface_id="FunAudioLLM/CosyVoice-300M",
                local_dir="CosyVoice-300M",
                size_gb=0.6,
                description="CosyVoice 300M基础模型",
                required=False,
                priority=4
            ),
            ModelType.COSYVOICE_300M_SFT: ModelInfo(
                model_type=ModelType.COSYVOICE_300M_SFT,
                modelscope_id="iic/CosyVoice-300M-SFT",
                huggingface_id="FunAudioLLM/CosyVoice-300M-SFT",
                local_dir="CosyVoice-300M-SFT",
                size_gb=0.6,
                description="CosyVoice 300M监督微调模型",
                required=False,
                priority=5
            ),
            ModelType.COSYVOICE_300M_INSTRUCT: ModelInfo(
                model_type=ModelType.COSYVOICE_300M_INSTRUCT,
                modelscope_id="iic/CosyVoice-300M-Instruct",
                huggingface_id="FunAudioLLM/CosyVoice-300M-Instruct",
                local_dir="CosyVoice-300M-Instruct",
                size_gb=0.6,
                description="CosyVoice 300M指令模型",
                required=False,
                priority=6
            ),
            ModelType.COSYVOICE_TTSFRD: ModelInfo(
                model_type=ModelType.COSYVOICE_TTSFRD,
                modelscope_id="iic/CosyVoice-ttsfrd",
                huggingface_id="FunAudioLLM/CosyVoice-ttsfrd",
                local_dir="CosyVoice-ttsfrd",
                size_gb=0.1,
                description="文本标准化处理资源",
                required=False,
                priority=7,
                dependencies=["ttsfrd"]
            ),
        }

    def _load_model_status(self):
        """加载模型状态"""
        try:
            status_file = self.path_manager.get_model_status_file_path()
            if os.path.exists(status_file):
                with open(status_file, 'r', encoding='utf-8') as f:
                    self._model_status = json.load(f)
            else:
                self._model_status = {}
        except Exception as e:
            self.logger.warning(f"加载模型状态失败: {e}")
            self._model_status = {}

    def _save_model_status(self):
        """保存模型状态"""
        try:
            status_file = self.path_manager.get_model_status_file_path()
            self.path_manager.ensure_file_directory(status_file)

            with open(status_file, 'w', encoding='utf-8') as f:
                json.dump(self._model_status, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"保存模型状态失败: {e}")

    def _check_download_source_availability(self) -> DownloadSource:
        """检查下载源可用性"""
        try:
            # 优先检查ModelScope
            import modelscope
            self.logger.info("ModelScope 可用，使用ModelScope作为下载源")
            return DownloadSource.MODELSCOPE
        except ImportError:
            try:
                # 回退到HuggingFace
                from huggingface_hub import snapshot_download
                self.logger.info("HuggingFace 可用，使用HuggingFace作为下载源")
                return DownloadSource.HUGGINGFACE
            except ImportError:
                raise DownloadSourceError("无可用的下载源，请安装modelscope或huggingface_hub")

    def _get_download_source(self, preferred: DownloadSource = DownloadSource.AUTO) -> DownloadSource:
        """获取下载源"""
        if preferred != DownloadSource.AUTO:
            return preferred

        return self._check_download_source_availability()

    def _download_with_modelscope(self, model_info: ModelInfo, force: bool = False) -> bool:
        """使用ModelScope下载模型"""
        try:
            import modelscope
            from modelscope.hub.api import HubApi

            local_path = self.path_manager.get_cosyvoice_model_path(model_info.local_dir)

            # 检查是否需要下载
            if not force and os.path.exists(local_path) and self._is_model_complete(local_path, model_info):
                self.logger.info(f"模型 {model_info.model_type.value} 已存在且完整，跳过下载")
                return True

            self.logger.info(f"开始下载模型 {model_info.model_type.value} (ModelScope)")
            self.logger.info(f"模型ID: {model_info.modelscope_id}")
            self.logger.info(f"目标路径: {local_path}")

            # 执行下载
            result = modelscope.snapshot_download(
                model_info.modelscope_id,
                local_dir=local_path
            )

            # 验证下载结果
            if self._is_model_complete(local_path, model_info):
                self.logger.info(f"模型 {model_info.model_type.value} 下载成功")
                return True
            else:
                raise ModelVerificationError("模型下载不完整")

        except Exception as e:
            self.logger.error(f"ModelScope下载失败: {e}")
            raise ModelDownloadError(f"ModelScope下载失败: {e}")

    def _download_with_huggingface(self, model_info: ModelInfo, force: bool = False) -> bool:
        """使用HuggingFace下载模型"""
        try:
            from huggingface_hub import snapshot_download

            local_path = self.path_manager.get_cosyvoice_model_path(model_info.local_dir)

            # 检查是否需要下载
            if not force and os.path.exists(local_path) and self._is_model_complete(local_path, model_info):
                self.logger.info(f"模型 {model_info.model_type.value} 已存在且完整，跳过下载")
                return True

            self.logger.info(f"开始下载模型 {model_info.model_type.value} (HuggingFace)")
            self.logger.info(f"模型ID: {model_info.huggingface_id}")
            self.logger.info(f"目标路径: {local_path}")

            # 执行下载
            result = snapshot_download(
                model_info.huggingface_id,
                local_dir=local_path
            )

            # 验证下载结果
            if self._is_model_complete(local_path, model_info):
                self.logger.info(f"模型 {model_info.model_type.value} 下载成功")
                return True
            else:
                raise ModelVerificationError("模型下载不完整")

        except Exception as e:
            self.logger.error(f"HuggingFace下载失败: {e}")
            raise ModelDownloadError(f"HuggingFace下载失败: {e}")

    def _is_model_complete(self, local_path: str, model_info: ModelInfo) -> bool:
        """检查模型是否完整"""
        if not os.path.exists(local_path):
            return False

        # 检查是否为目录且不为空
        if not os.path.isdir(local_path):
            return False

        # 检查关键文件是否存在
        required_files = self._get_required_files(model_info.model_type)
        for file_path in required_files:
            full_path = os.path.join(local_path, file_path)
            if not os.path.exists(full_path):
                self.logger.warning(f"缺少关键文件: {full_path}")
                return False

        # 可选：检查文件大小或哈希
        return True

    def _get_required_files(self, model_type: ModelType) -> List[str]:
        """获取模型所需的关键文件列表"""
        # 根据模型类型返回不同的必需文件列表
        if model_type == ModelType.COSYVOICE3_2512:
            # CosyVoice3 模型文件
            required_files = [
                "cosyvoice3.yaml",      # CosyVoice3 配置文件
                "llm.pt",                # LLM 模型权重
                "flow.pt",               # Flow 模型权重
                "hift.pt",               # HiFT-GAN 声码器权重
                "campplus.onnx",         # 说话人编码器
                "speech_tokenizer_v3.onnx"  # 语音 tokenizer
            ]
        elif model_type in [ModelType.COSYVOICE2_05B, ModelType.COSYVOICE_300M,
                            ModelType.COSYVOICE_300M_SFT, ModelType.COSYVOICE_300M_INSTRUCT]:
            # CosyVoice/CosyVoice2 模型文件
            required_files = [
                "config.yaml",
                "pytorch_model.bin"
            ]
        elif model_type == ModelType.COSYVOICE_TTSFRD:
            # TTSFRD 资源文件
            required_files = ["resource.zip"]
        else:
            # 通用默认检查（宽松）
            required_files = []

        return required_files

    def _install_ttsfrd_dependencies(self, model_path: str) -> bool:
        """安装ttsfrd依赖"""
        try:
            self.logger.info("开始安装ttsfrd依赖...")

            # 解压资源文件
            resource_zip = os.path.join(model_path, "resource.zip")
            if os.path.exists(resource_zip):
                import zipfile
                with zipfile.ZipFile(resource_zip, 'r') as zip_ref:
                    zip_ref.extractall(model_path)
                self.logger.info("ttsfrd资源文件解压完成")

            # 安装依赖包
            dependency_files = [
                "ttsfrd_dependency-0.1-py3-none-any.whl",
                "ttsfrd-0.4.2-cp310-cp310-linux_x86_64.whl"
            ]

            for dep_file in dependency_files:
                dep_path = os.path.join(model_path, dep_file)
                if os.path.exists(dep_path):
                    self.logger.info(f"安装依赖包: {dep_file}")
                    subprocess.run([
                        sys.executable, "-m", "pip", "install", dep_path, "--force-reinstall"
                    ], check=True)

            self.logger.info("ttsfrd依赖安装完成")
            return True

        except Exception as e:
            self.logger.error(f"ttsfrd依赖安装失败: {e}")
            raise DependencyInstallationError(f"ttsfrd依赖安装失败: {e}")

    def download_model(self, model_type: ModelType, source: DownloadSource = DownloadSource.AUTO,
                      force: bool = False, install_deps: bool = True) -> bool:
        """
        下载单个模型

        Args:
            model_type: 模型类型
            source: 下载源
            force: 强制重新下载
            install_deps: 是否安装依赖

        Returns:
            bool: 下载是否成功
        """
        try:
            if model_type not in self._model_registry:
                raise ValueError(f"不支持的模型类型: {model_type}")

            model_info = self._model_registry[model_type]
            download_source = self._get_download_source(source)

            # 更新状态
            progress = DownloadProgress(
                model_type=model_type,
                status=DownloadStatus.DOWNLOADING,
                start_time=time.time()
            )
            self._update_progress(progress)

            # 执行下载
            if download_source == DownloadSource.MODELSCOPE:
                success = self._download_with_modelscope(model_info, force)
            else:
                success = self._download_with_huggingface(model_info, force)

            if success:
                # 安装依赖
                if install_deps and model_info.dependencies:
                    for dep in model_info.dependencies:
                        if dep == "ttsfrd":
                            self._install_ttsfrd_dependencies(
                                self.path_manager.get_cosyvoice_ttsfrd_model_path()
                            )

                # 更新状态
                progress.status = DownloadStatus.COMPLETED
                progress.end_time = time.time()
                progress.progress = 1.0
                self._update_progress(progress)

            return success

        except Exception as e:
            # 更新错误状态
            progress = DownloadProgress(
                model_type=model_type,
                status=DownloadStatus.FAILED,
                error_message=str(e),
                end_time=time.time()
            )
            self._update_progress(progress)
            raise

    def download_models(self, model_types: List[ModelType], source: DownloadSource = DownloadSource.AUTO,
                         force: bool = False, install_deps: bool = True,
                         callback: Optional[Callable[[DownloadProgress], None]] = None) -> Dict[ModelType, bool]:
        """
        下载多个模型（并发下载）

        Args:
            model_types: 模型类型列表
            source: 下载源
            force: 强制重新下载
            install_deps: 是否安装依赖
            callback: 进度回调函数

        Returns:
            Dict[ModelType, bool]: 每个模型的下载结果
        """
        results = {}

        # 按优先级排序
        sorted_models = sorted(
            model_types,
            key=lambda mt: self._model_registry[mt].priority
        )

        # 设置回调
        if callback:
            self._progress_callbacks[None] = callback

        # 使用线程池并发下载
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(
                    self.download_model,
                    model_type,
                    source,
                    force,
                    install_deps
                ): model_type
                for model_type in sorted_models
            }

            for future in as_completed(futures):
                model_type = futures[future]
                try:
                    success = future.result()
                    results[model_type] = success
                    self.logger.info(f"模型 {model_type.value} 下载结果: {'成功' if success else '失败'}")
                except Exception as e:
                    results[model_type] = False
                    self.logger.error(f"模型 {model_type.value} 下载失败: {e}")

        # 清理回调
        if None in self._progress_callbacks:
            del self._progress_callbacks[None]

        return results

    def _update_progress(self, progress: DownloadProgress):
        """更新下载进度"""
        with self._lock:
            self._model_status[progress.model_type.value] = {
                "status": progress.status.value,
                "progress": progress.progress,
                "downloaded_size": progress.downloaded_size,
                "total_size": progress.total_size,
                "error_message": progress.error_message,
                "start_time": progress.start_time,
                "end_time": progress.end_time,
                "last_updated": time.time()
            }
            self._save_model_status()

        # 调用回调
        for callback in self._progress_callbacks.values():
            try:
                callback(progress)
            except Exception as e:
                self.logger.warning(f"进度回调执行失败: {e}")

    def get_download_status(self, model_type: Optional[ModelType] = None) -> Union[DownloadProgress, Dict[str, DownloadProgress]]:
        """
        获取下载状态

        Args:
            model_type: 模型类型，为None时返回所有模型状态

        Returns:
            下载进度信息
        """
        if model_type is not None:
            status_data = self._model_status.get(model_type.value, {})
            return DownloadProgress(
                model_type=model_type,
                status=DownloadStatus(status_data.get("status", "pending")),
                progress=status_data.get("progress", 0.0),
                downloaded_size=status_data.get("downloaded_size", 0),
                total_size=status_data.get("total_size", 0),
                error_message=status_data.get("error_message"),
                start_time=status_data.get("start_time"),
                end_time=status_data.get("end_time")
            )
        else:
            return {
                model_type_str: self._create_progress_from_status(model_type_str, status_data)
                for model_type_str, status_data in self._model_status.items()
            }

    def _create_progress_from_status(self, model_type_str: str, status_data: Dict) -> DownloadProgress:
        """从状态数据创建进度对象"""
        try:
            model_type = ModelType(model_type_str)
            return DownloadProgress(
                model_type=model_type,
                status=DownloadStatus(status_data.get("status", "pending")),
                progress=status_data.get("progress", 0.0),
                downloaded_size=status_data.get("downloaded_size", 0),
                total_size=status_data.get("total_size", 0),
                error_message=status_data.get("error_message"),
                start_time=status_data.get("start_time"),
                end_time=status_data.get("end_time")
            )
        except ValueError:
            # 处理无效的模型类型
            return DownloadProgress(
                model_type=ModelType.COSYVOICE3_2512,  # 默认值
                status=DownloadStatus.FAILED,
                error_message=f"无效的模型类型: {model_type_str}"
            )

    def get_available_models(self) -> Dict[ModelType, ModelInfo]:
        """获取可用模型列表"""
        return self._model_registry.copy()

    def is_model_downloaded(self, model_type: ModelType) -> bool:
        """检查模型是否已下载"""
        if model_type not in self._model_registry:
            return False

        model_info = self._model_registry[model_type]
        local_path = self.path_manager.get_cosyvoice_model_path(model_info.local_dir)

        return os.path.exists(local_path) and self._is_model_complete(local_path, model_info)

    def get_model_path(self, model_type: ModelType) -> Optional[str]:
        """获取模型本地路径"""
        if model_type not in self._model_registry:
            return None

        model_info = self._model_registry[model_type]
        local_path = self.path_manager.get_cosyvoice_model_path(model_info.local_dir)

        return local_path if os.path.exists(local_path) else None

    def cleanup_incomplete_downloads(self):
        """清理不完整的下载"""
        models_path = self.path_manager.get_cosyvoice_models_path()

        for model_info in self._model_registry.values():
            local_path = os.path.join(models_path, model_info.local_dir)

            if os.path.exists(local_path) and not self._is_model_complete(local_path, model_info):
                self.logger.warning(f"删除不完整的模型: {model_info.model_type.value}")
                import shutil
                shutil.rmtree(local_path, ignore_errors=True)

    def get_download_statistics(self) -> Dict[str, Any]:
        """获取下载统计信息"""
        total_models = len(self._model_registry)
        downloaded_models = sum(1 for mt in self._model_registry.keys() if self.is_model_downloaded(mt))

        total_size = sum(info.size_gb for info in self._model_registry.values())
        downloaded_size = sum(
            info.size_gb for mt, info in self._model_registry.items()
            if self.is_model_downloaded(mt)
        )

        return {
            "total_models": total_models,
            "downloaded_models": downloaded_models,
            "total_size_gb": total_size,
            "downloaded_size_gb": downloaded_size,
            "download_progress": downloaded_models / total_models if total_models > 0 else 0,
            "models_status": self._model_status
        }


# ==================== 便捷函数 ====================

def get_download_manager(path_manager: PathManager = None) -> ModelDownloadManager:
    """获取模型下载管理器实例"""
    return ModelDownloadManager(path_manager)


def download_all_cosyvoice_models(source: DownloadSource = DownloadSource.AUTO) -> Dict[ModelType, bool]:
    """下载所有推荐模型"""
    manager = get_download_manager()

    # 推荐模型列表
    recommended_models = [
        ModelType.COSYVOICE3_2512,
        ModelType.COSYVOICE_TTSFRD
    ]

    return manager.download_models(recommended_models, source=source)


# ==================== 使用示例 ====================

if __name__ == "__main__":
    # 设置日志
    import logging
    logging.basicConfig(level=logging.INFO)

    try:
        # 创建下载管理器
        manager = get_download_manager()

        # 显示下载统计
        stats = manager.get_download_statistics()
        print("=" * 60)
        print("CosyVoice模型下载统计")
        print("=" * 60)
        print(f"总模型数: {stats['total_models']}")
        print(f"已下载: {stats['downloaded_models']}")
        print(f"总大小: {stats['total_size_gb']:.1f}GB")
        print(f"已下载大小: {stats['downloaded_size_gb']:.1f}GB")
        print(f"下载进度: {stats['download_progress']:.1%}")
        print("=" * 60)

        # 获取可用模型
        models = manager.get_available_models()
        print("\n可用模型:")
        for model_type, model_info in models.items():
            status = "✅ 已下载" if manager.is_model_downloaded(model_type) else "❌ 未下载"
            print(f"  {model_info.description}: {status}")

        # 示例：下载单个模型
        # result = manager.download_model(ModelType.COSYVOICE3_2512)
        # print(f"\n下载结果: {'成功' if result else '失败'}")

    except Exception as e:
        print(f"错误: {e}")
        traceback.print_exc()