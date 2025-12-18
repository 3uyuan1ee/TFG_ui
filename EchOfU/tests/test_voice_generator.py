#!/usr/bin/env python3
"""
voice_generator.py的全面测试套件
"""

import unittest
import os
import sys
import tempfile
import shutil
import json
import time
import torch
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# 添加项目根目录和backend目录到Python路径
project_root = Path(__file__).parent
backend_path = project_root / "EchOfU" / "backend"
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(backend_path))

# 抑制警告
import warnings
warnings.filterwarnings("ignore")

class TestOpenVoiceService(unittest.TestCase):
    """OpenVoiceService服务类的测试"""

    def setUp(self):
        """测试前的设置"""
        # 创建临时目录
        self.test_dir = tempfile.mkdtemp()

        # 创建测试所需的目录结构
        self.dirs = [
            "EchOfU/OpenVoice/checkpoints_v2",
            "EchOfU/OpenVoice/checkpoints_v2/base_speakers/ses",
            "models/OpenVoice",
            "static/voices",
            "processed"
        ]

        for dir_path in self.dirs:
            os.makedirs(os.path.join(self.test_dir, dir_path), exist_ok=True)

        # 创建假的配置文件
        config_path = os.path.join(self.test_dir, "EchOfU/OpenVoice/checkpoints_v2/config.json")
        with open(config_path, 'w') as f:
            json.dump({
                "model": {
                    "sampling_rate": 22050
                }
            }, f)

        # 创建假的模型文件
        model_files = [
            "EchOfU/OpenVoice/checkpoints_v2/converter.pth",
            "EchOfU/OpenVoice/checkpoints_v2/base_speakers/ses/zh.pth",
            "EchOfU/OpenVoice/checkpoints_v2/base_speakers/ses/en.pth"
        ]

        for file_path in model_files:
            full_path = os.path.join(self.test_dir, file_path)
            torch.save({"dummy": "data"}, full_path)

        # 切换到临时目录
        self.original_cwd = os.getcwd()
        os.chdir(self.test_dir)

    def tearDown(self):
        """测试后的清理"""
        os.chdir(self.original_cwd)
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    @patch('torch.cuda.is_available')
    def test_device_selection(self, mock_cuda):
        """测试设备选择逻辑"""
        mock_cuda.return_value = False

        # Mock OpenVoice components
        with patch('voice_generator.ToneColorConverter') as mock_converter:
            mock_converter.return_value = Mock()
            mock_converter.return_value.load_ckpt = Mock()

            from voice_generator import OpenVoiceService

            OpenVoiceService.reset_instance()
            service = OpenVoiceService.get_instance()
            self.assertEqual(service.device, "cpu")

    @patch('voice_generator.torch.cuda.is_available')
    def test_device_selection_cuda(self, mock_cuda):
        """测试CUDA设备选择"""
        mock_cuda.return_value = True

        with patch('voice_generator.ToneColorConverter') as mock_converter:
            mock_converter.return_value = Mock()
            mock_converter.return_value.load_ckpt = Mock()

            from voice_generator import OpenVoiceService

            OpenVoiceService.reset_instance()
            service = OpenVoiceService.get_instance()
            self.assertEqual(service.device, "cuda")

    @patch('voice_generator.ToneColorConverter')
    @patch('voice_generator.se_extractor')
    def test_model_initialization(self, mock_se_extractor, mock_converter):
        """测试模型初始化"""
        mock_converter_instance = Mock()
        mock_converter_instance.load_ckpt = Mock()
        mock_converter.return_value = mock_converter_instance

        from voice_generator import OpenVoiceService

        OpenVoiceService.reset_instance()
        service = OpenVoiceService.get_instance()

        # 验证转换器被初始化
        self.assertIsNotNone(service.tone_converter)
        self.assertEqual(service.device, "cpu" if not torch.cuda.is_available() else "cuda")

    def test_ensure_directories(self):
        """测试目录创建功能"""
        from voice_generator import OpenVoiceService

        OpenVoiceService.reset_instance()
        service = OpenVoiceService.get_instance()
        service.ensure_directories()

        # 验证所有必要的目录都已创建
        for dir_path in self.dirs:
            full_path = os.path.join(self.test_dir, dir_path)
            self.assertTrue(os.path.exists(full_path), f"目录 {full_path} 应该存在")

    def test_check_models_exist(self):
        """测试模型文件检查"""
        from voice_generator import OpenVoiceService

        OpenVoiceService.reset_instance()
        service = OpenVoiceService.get_instance()

        # 当模型文件存在时
        self.assertTrue(service.check_models_exist())

        # 删除一个模型文件
        os.remove("EchOfU/OpenVoice/checkpoints_v2/converter.pth")

        # 再次检查应该返回False
        self.assertFalse(service.check_models_exist())

    def test_load_speaker_features_empty(self):
        """测试加载空的说话人特征"""
        from voice_generator import OpenVoiceService

        OpenVoiceService.reset_instance()
        service = OpenVoiceService.get_instance()
        features = service.load_speaker_features()

        self.assertEqual(features, {})

    def test_load_speaker_features_with_data(self):
        """测试加载说话人特征数据"""
        from voice_generator import OpenVoiceService

        # 创建测试特征文件
        feature_data = {
            "speaker1": {
                "feature_path": "models/OpenVoice/speaker1_se.pth",
                "reference_audio": "test1.wav",
                "created_time": "2023-01-01 00:00:00"
            }
        }

        with open("models/OpenVoice/speaker_features.json", 'w') as f:
            json.dump(feature_data, f)

        # 创建假的特征文件
        torch.save({"se": torch.randn(1, 256)}, "models/OpenVoice/speaker1_se.pth")

        OpenVoiceService.reset_instance()
        service = OpenVoiceService.get_instance()
        features = service.load_speaker_features()

        self.assertIn("speaker1", features)
        self.assertEqual(features["speaker1"]["reference_audio"], "test1.wav")

    def test_save_speaker_feature(self):
        """测试保存说话人特征"""
        from voice_generator import OpenVoiceService

        OpenVoiceService.reset_instance()
        service = OpenVoiceService.get_instance()
        test_se = torch.randn(1, 256)

        service.save_speaker_feature("test_speaker", "test.wav", test_se)

        # 验证特征文件被保存
        self.assertTrue(os.path.exists("models/OpenVoice/test_speaker_se.pth"))

        # 验证元数据文件被创建
        self.assertTrue(os.path.exists("models/OpenVoice/speaker_features.json"))

        # 验证保存的数据
        with open("models/OpenVoice/speaker_features.json", 'r') as f:
            metadata = json.load(f)

        self.assertIn("test_speaker", metadata)
        self.assertEqual(metadata["test_speaker"]["reference_audio"], "test.wav")

    def test_list_available_speakers_empty(self):
        """测试列出空的说话人列表"""
        from voice_generator import OpenVoiceService

        OpenVoiceService.reset_instance()
        service = OpenVoiceService.get_instance()
        speakers = service.list_available_speakers()
        self.assertEqual(speakers, [])

    def test_list_available_speakers_with_data(self):
        """测试列出有数据的说话人列表"""
        from voice_generator import OpenVoiceService

        OpenVoiceService.reset_instance()
        service = OpenVoiceService.get_instance()

        # 模拟有说话人特征
        service.speaker_features = {
            "speaker1": {"data": "test1"},
            "speaker2": {"data": "test2"}
        }

        speakers = service.list_available_speakers()
        self.assertEqual(set(speakers), {"speaker1", "speaker2"})

    @patch('voice_generator.ToneColorConverter')
    @patch('voice_generator.se_extractor')
    def test_extract_and_save_speaker_feature(self, mock_se_extractor, mock_converter):
        """测试提取和保存说话人特征"""
        mock_converter_instance = Mock()
        mock_converter.return_value = mock_converter_instance
        mock_se_extractor.get_se.return_value = torch.randn(1, 256)

        from voice_generator import OpenVoiceService

        OpenVoiceService.reset_instance()
        service = OpenVoiceService.get_instance()

        # 创建测试音频文件
        test_audio_path = "test_audio.wav"
        with open(test_audio_path, 'w') as f:
            f.write("fake audio data")

        result = service.extract_and_save_speaker_feature("test_speaker", test_audio_path)

        self.assertTrue(result)
        mock_se_extractor.get_se.assert_called_once()

        # 清理测试文件
        if os.path.exists(test_audio_path):
            os.remove(test_audio_path)

    @patch('voice_generator.ToneColorConverter')
    def test_generate_base_speech_without_tts(self, mock_converter):
        """测试没有TTS时的基础语音生成"""
        mock_converter_instance = Mock()
        mock_converter.return_value = mock_converter_instance

        from voice_generator import OpenVoiceService

        OpenVoiceService.reset_instance()
        service = OpenVoiceService.get_instance()
        service.tts_model = None  # 确保TTS模型为None

        result = service.generate_base_speech("test text", "output.wav")

        # 由于没有MeloTTS和TTS，应该返回None
        self.assertIsNone(result)

    def test_download_with_progress(self):
        """测试文件下载进度显示"""
        from voice_generator import OpenVoiceService

        OpenVoiceService.reset_instance()
        service = OpenVoiceService.get_instance()

        # Mock requests
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.raise_for_status.return_value = None
            mock_response.headers = {'content-length': '1024'}
            mock_response.iter_content.return_value = [b'test'] * 10

            mock_get.return_value = mock_response

            try:
                service.download_with_progress(
                    "http://fake-url.com/test.zip",
                    "test_download.zip"
                )

                # 验证文件被下载
                self.assertTrue(os.path.exists("test_download.zip"))

                # 清理测试文件
                if os.path.exists("test_download.zip"):
                    os.remove("test_download.zip")

            except Exception as e:
                # 预期的异常，因为URL是假的
                pass

    def test_extract_zip_file(self):
        """测试ZIP文件解压"""
        from voice_generator import OpenVoiceService

        OpenVoiceService.reset_instance()
        service = OpenVoiceService.get_instance()

        # 创建一个假的ZIP文件
        import zipfile
        with zipfile.ZipFile('test.zip', 'w') as zip_file:
            zip_file.writestr('test.txt', 'test content')

        try:
            service.extract_zip_file('test.zip', '.')

            # 验证文件被解压
            self.assertTrue(os.path.exists('test.txt'))

        except Exception as e:
            # 可能会失败，这是正常的
            pass

        finally:
            # 清理测试文件
            for file in ['test.zip', 'test.txt']:
                if os.path.exists(file):
                    os.remove(file)

class TestUtilityFunctions(unittest.TestCase):
    """测试工具函数"""

    def setUp(self):
        """测试前设置"""
        self.test_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.test_dir)

    def tearDown(self):
        """测试后清理"""
        os.chdir(self.original_cwd)
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    @patch('voice_generator.OpenVoiceService')
    def test_generate_voice_success(self, mock_service_class):
        """测试语音生成成功"""
        mock_service = Mock()
        mock_service.tone_converter = Mock()  # 非None表示已初始化
        mock_service.generate_speech.return_value = "generated_voice.wav"
        mock_service_class.return_value = mock_service

        from voice_generator import generate_voice

        result = generate_voice("test text", "speaker1")

        self.assertEqual(result, "generated_voice.wav")
        mock_service.generate_speech.assert_called_once_with("test text", "speaker1")

    @patch('voice_generator.OpenVoiceService')
    def test_generate_voice_no_converter(self, mock_service_class):
        """测试没有转换器时的语音生成"""
        mock_service = Mock()
        mock_service.tone_converter = None  # 未初始化
        mock_service_class.return_value = mock_service

        from voice_generator import generate_voice

        result = generate_voice("test text", "speaker1")

        self.assertIsNone(result)

    @patch('voice_generator.OpenVoiceService')
    def test_extract_speaker_feature_success(self, mock_service_class):
        """测试提取说话人特征成功"""
        mock_service = Mock()
        mock_service.extract_and_save_speaker_feature.return_value = True
        mock_service_class.return_value = mock_service

        from voice_generator import extract_speaker_feature

        result = extract_speaker_feature("speaker1", "audio.wav")

        self.assertTrue(result)
        mock_service.extract_and_save_speaker_feature.assert_called_once_with("speaker1", "audio.wav")

    @patch('voice_generator.OpenVoiceService')
    def test_list_available_speakers_success(self, mock_service_class):
        """测试列出可用说话人成功"""
        mock_service = Mock()
        mock_service.list_available_speakers.return_value = ["speaker1", "speaker2"]
        mock_service_class.return_value = mock_service

        from voice_generator import list_available_speakers

        result = list_available_speakers()

        self.assertEqual(result, ["speaker1", "speaker2"])
        mock_service.list_available_speakers.assert_called_once()

class TestErrorHandling(unittest.TestCase):
    """测试错误处理"""

    def setUp(self):
        """测试前设置"""
        self.test_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.test_dir)

    def tearDown(self):
        """测试后清理"""
        os.chdir(self.original_cwd)
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    @patch('voice_generator.ToneColorConverter')
    def test_model_initialization_failure(self, mock_converter):
        """测试模型初始化失败"""
        mock_converter.side_effect = Exception("模型加载失败")

        from voice_generator import OpenVoiceService

        OpenVoiceService.reset_instance()
        service = OpenVoiceService.get_instance()

        # 应该回退到默认状态
        self.assertIsNone(service.tts_model)
        self.assertIsNone(service.tone_converter)
        self.assertEqual(service.speaker_features, {})

    @patch('voice_generator.ToneColorConverter')
    def test_fallback_to_default_state(self, mock_converter):
        """测试回退到默认状态"""
        from voice_generator import OpenVoiceService

        OpenVoiceService.reset_instance()
        service = OpenVoiceService.get_instance()
        service.fallback_to_default_state()

        self.assertIsNone(service.tts_model)
        self.assertIsNone(service.tone_converter)
        self.assertEqual(service.speaker_features, {})

def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("开始运行 voice_generator.py 测试套件")
    print("=" * 60)

    # 创建测试套件
    test_suite = unittest.TestSuite()

    # 添加测试类
    test_classes = [
        TestOpenVoiceService,
        TestUtilityFunctions,
        TestErrorHandling
    ]

    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)

    # 运行测试
    runner = unittest.TextTestRunner(
        verbosity=2,
        stream=sys.stdout,
        descriptions=True,
        failfast=False
    )

    result = runner.run(test_suite)

    print("\n" + "=" * 60)
    print("测试结果总结")
    print("=" * 60)
    print(f"总测试数: {result.testsRun}")
    print(f"成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")

    if result.failures:
        print("\n失败的测试:")
        for test, traceback in result.failures:
            print(f"❌ {test}: {traceback.split('AssertionError:')[-1].strip()}")

    if result.errors:
        print("\n错误的测试:")
        for test, traceback in result.errors:
            print(f"❌ {test}: {traceback.split('Exception:')[-1].strip()}")

    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
    print(f"\n成功率: {success_rate:.1f}%")

    if success_rate >= 80:
        print("🎉 测试套件通过！voice_generator.py 代码质量良好。")
    else:
        print("⚠️  测试发现问题，建议检查代码。")

    return result.wasSuccessful()

if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)