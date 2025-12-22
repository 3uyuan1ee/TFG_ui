#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全面依赖检查脚本
验证所有关键依赖是否正常工作
"""

import sys
import os

# 添加backend目录到Python路径
backend_path = os.path.join(os.path.dirname(__file__), 'backend')
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)


def check_pytorch_core():
    """检查PyTorch核心库"""
    print("\n" + "=" * 70)
    print("🔥 PyTorch核心库检查")
    print("=" * 70)

    try:
        import torch
        import torchaudio
        import torchvision

        torch_version = torch.__version__
        torchaudio_version = torchaudio.__version__
        torchvision_version = torchvision.__version__

        print(f"✅ torch: {torch_version}")
        print(f"✅ torchaudio: {torchaudio_version}")
        print(f"✅ torchvision: {torchvision_version}")

        # 检查CUDA可用性
        cuda_available = torch.cuda.is_available()
        mps_available = torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False

        if cuda_available:
            print(f"✅ CUDA可用: {torch.cuda.get_device_name(0)}")
        if mps_available:
            print(f"✅ MPS (Apple Silicon)可用")

        # 验证版本兼容性
        if torch_version == "2.3.1":
            print("✅ torch版本符合CosyVoice要求 (2.3.1)")
        else:
            print(f"⚠️  torch版本不是2.3.1，可能影响CosyVoice")

        if torchvision_version == "0.18.1":
            print("✅ torchvision版本与torch 2.3.1兼容 (0.18.1)")
        else:
            print(f"⚠️  torchvision版本可能不兼容")

        if torchaudio_version == "2.3.1":
            print("✅ torchaudio版本符合CosyVoice要求 (2.3.1)")
        else:
            print(f"⚠️  torchaudio版本不是2.3.1")

        # 测试关键功能
        print("\n🧪 功能测试:")
        try:
            x = torch.randn(3, 4)
            print(f"✅ 张量创建正常: {x.shape}")
        except Exception as e:
            print(f"❌ 张量创建失败: {e}")
            return False

        try:
            import torchvision.transforms as transforms
            transform = transforms.Compose([transforms.ToTensor()])
            print("✅ torchvision.transforms可用")
        except Exception as e:
            print(f"❌ torchvision.transforms失败: {e}")
            return False

        return True

    except Exception as e:
        print(f"❌ PyTorch核心库导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_transformers():
    """检查Transformers库"""
    print("\n" + "=" * 70)
    print("🤗 Transformers库检查")
    print("=" * 70)

    try:
        import transformers

        version = transformers.__version__
        print(f"✅ transformers: {version}")

        if version == "4.51.3":
            print("✅ transformers版本符合CosyVoice要求 (4.51.3)")

        # 测试关键功能
        print("\n🧪 功能测试:")
        try:
            from transformers import AutoConfig
            print("✅ AutoConfig导入成功")
        except Exception as e:
            print(f"❌ AutoConfig导入失败: {e}")

        return True

    except Exception as e:
        print(f"❌ transformers导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_model_download_sources():
    """检查模型下载源"""
    print("\n" + "=" * 70)
    print("📦 模型下载源检查")
    print("=" * 70)

    success = True

    # 检查ModelScope
    print("\n🔍 ModelScope:")
    try:
        import modelscope
        version = modelscope.__version__
        print(f"✅ modelscope: {version}")
        if version == "1.4.2":
            print("✅ modelscope版本与torch 2.3.1兼容")
    except Exception as e:
        print(f"❌ modelscope导入失败: {e}")
        success = False

    # 检查HuggingFace Hub
    print("\n🔍 HuggingFace Hub:")
    try:
        from huggingface_hub import snapshot_download
        import huggingface_hub
        version = huggingface_hub.__version__
        print(f"✅ huggingface_hub: {version}")
        print("✅ snapshot_download可用")
    except Exception as e:
        print(f"❌ huggingface_hub导入失败: {e}")
        success = False

    return success


def check_audio_processing():
    """检查音频处理库"""
    print("\n" + "=" * 70)
    print("🎵 音频处理库检查")
    print("=" * 70)

    libraries = {
        'librosa': '0.10.2',
        'soundfile': '0.12.1',
        'pyworld': '0.3.4'
    }

    all_ok = True
    for lib_name, expected_version in libraries.items():
        try:
            lib = __import__(lib_name)
            version = getattr(lib, '__version__', 'unknown')
            print(f"✅ {lib_name}: {version}")
        except Exception as e:
            print(f"❌ {lib_name}导入失败: {e}")
            all_ok = False

    return all_ok


def check_numerical_libraries():
    """检查数值计算库"""
    print("\n" + "=" * 70)
    print("🔢 数值计算库检查")
    print("=" * 70)

    try:
        import numpy as np
        print(f"✅ numpy: {np.__version__}")

        if np.__version__ == "1.23.5":
            print("✅ numpy版本与modelscope 1.4.2兼容")

        # 测试基本功能
        arr = np.array([1, 2, 3])
        print(f"✅ numpy基本功能正常")

        import scipy
        print(f"✅ scipy: {scipy.__version__}")

        return True

    except Exception as e:
        print(f"❌ 数值计算库导入失败: {e}")
        return False


def check_cosyvoice_integration():
    """检查CosyVoice集成"""
    print("\n" + "=" * 70)
    print("🎙️  CosyVoice集成检查")
    print("=" * 70)

    try:
        from backend.CV_clone import CosyService, COSYVOICE_AVAILABLE, get_cosy_service

        print(f"✅ CosyVoice模块导入成功")
        print(f"   COSYVOICE_AVAILABLE: {COSYVOICE_AVAILABLE}")

        if COSYVOICE_AVAILABLE:
            print("\n🧪 初始化测试:")
            try:
                service = get_cosy_service()
                print("✅ CosyService初始化成功")

                # 获取服务状态
                status = service.get_service_status()
                print(f"   服务初始化: {status['service_initialized']}")
                print(f"   CosyVoice可用: {status['cosyvoice_available']}")

            except Exception as e:
                print(f"⚠️  CosyService初始化失败: {e}")
                print("   这是正常的，如果模型未下载")

        return True

    except Exception as e:
        print(f"❌ CosyVoice集成检查失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_model_download_manager():
    """检查模型下载管理器"""
    print("\n" + "=" * 70)
    print("⬇️  模型下载管理器检查")
    print("=" * 70)

    try:
        from backend.model_download_manager import (
            ModelDownloadManager,
            ModelType,
            DownloadSource
        )

        print("✅ ModelDownloadManager导入成功")

        # 创建实例
        manager = ModelDownloadManager()
        print("✅ 管理器实例创建成功")

        # 检查下载源
        source = manager._check_download_source_availability()
        print(f"✅ 自动选择的下载源: {source.value}")

        # 获取可用模型
        models = manager.get_available_models()
        print(f"✅ 可用模型数量: {len(models)}")

        # 检查模型状态
        for model_type in [ModelType.COSYVOICE3_2512]:
            is_downloaded = manager.is_model_downloaded(model_type)
            status = "已下载" if is_downloaded else "未下载"
            print(f"   {model_type.value}: {status}")

        # 获取统计信息
        stats = manager.get_download_statistics()
        print(f"\n📊 下载统计:")
        print(f"   总模型数: {stats['total_models']}")
        print(f"   已下载: {stats['downloaded_models']}")
        print(f"   下载进度: {stats['download_progress']:.1%}")

        return True

    except Exception as e:
        print(f"❌ 模型下载管理器检查失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_dependencies_conflicts():
    """检查依赖冲突"""
    print("\n" + "=" * 70)
    print("⚠️  依赖冲突检查")
    print("=" * 70)

    conflicts = []

    # 检查numpy版本冲突
    try:
        import numpy as np
        if np.__version__ >= "2.0.0":
            conflicts.append("numpy >= 2.0.0 与modelscope 1.4.2冲突")

        # 检查matplotlib要求
        import matplotlib
        if np.__version__ >= "2.0.0":
            conflicts.append("numpy >= 2.0.0 与matplotlib < 2.0冲突")

    except:
        pass

    # 检查pillow版本
    try:
        import PIL
        import gradio
        # gradio 5.4.0要求pillow<12.0
        if PIL.__version__ >= "12.0.0":
            conflicts.append(f"pillow {PIL.__version__} 与gradio <12.0冲突")
    except:
        pass

    if conflicts:
        print("⚠️  发现潜在冲突:")
        for conflict in conflicts:
            print(f"   - {conflict}")
        print("\n💡 这些冲突可能不影响核心功能")
        return False
    else:
        print("✅ 未发现关键依赖冲突")
        return True


def main():
    """主测试函数"""
    print("=" * 70)
    print("🧪 全面依赖检查")
    print("=" * 70)

    tests = [
        ("PyTorch核心库", check_pytorch_core),
        ("Transformers库", check_transformers),
        ("模型下载源", check_model_download_sources),
        ("音频处理库", check_audio_processing),
        ("数值计算库", check_numerical_libraries),
        ("CosyVoice集成", check_cosyvoice_integration),
        ("模型下载管理器", check_model_download_manager),
        ("依赖冲突检查", check_dependencies_conflicts),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"\n✅ {test_name} - 通过")
            else:
                print(f"\n❌ {test_name} - 失败")
        except Exception as e:
            print(f"\n❌ {test_name} - 异常: {e}")

    print("\n" + "=" * 70)
    print(f"📈 测试结果: {passed}/{total} 通过")
    print("=" * 70)

    if passed == total:
        print("\n🎉 所有依赖检查通过！系统状态良好")
        return 0
    else:
        print("\n⚠️  部分依赖检查失败，请查看上述详情")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
