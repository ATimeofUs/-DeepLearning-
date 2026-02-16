#!/usr/bin/env python3
import subprocess
from pathlib import Path
from PyQt6 import QtWidgets, QtGui
from PyQt6.QtCore import Qt, QThread, pyqtSignal
import sys


class UpscaleWorker(QThread):
    """后台线程处理图像放大任务"""

    progress = pyqtSignal(str)
    finished = pyqtSignal(bool, str)

    def __init__(
        self,
        input_path: str,
        output_path: str,
        scale: int,
        model: str,
        tile_size: str,
        gpu_id: str,
        tta: bool,
        format: str,
    ):
        super().__init__()
        self.input_path = input_path
        self.output_path = output_path
        self.scale = scale
        self.model = model
        self.tile_size = tile_size
        self.gpu_id = gpu_id
        self.tta = tta
        self.format = format

    def run(self):
        try:
            # 构建命令
            cmd = [
                "realesrgan-ncnn-vulkan",
                "-i",
                self.input_path,
                "-o",
                self.output_path,
                "-s",
                str(self.scale),
                "-n",
                self.model,
                "-t",
                self.tile_size,
                "-g",
                self.gpu_id,
                "-f",
                self.format,
            ]

            if self.tta:
                cmd.append("-x")

            self.progress.emit(f"执行命令: {' '.join(cmd)}")

            # 执行命令
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
            )

            # 实时输出
            for line in process.stdout:
                self.progress.emit(line.strip())

            process.wait()

            if process.returncode == 0:
                self.finished.emit(True, f"✅ 放大完成！输出文件: {self.output_path}")
            else:
                self.finished.emit(False, f"❌ 处理失败，返回码: {process.returncode}")

        except Exception as e:
            self.finished.emit(False, f"❌ 错误: {str(e)}")


class RealESRGANGui(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.worker = None
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Real-ESRGAN 图像放大工具")
        self.setMinimumWidth(700)

        layout = QtWidgets.QVBoxLayout()

        # === 输入文件选择 ===
        input_group = QtWidgets.QGroupBox("输入文件")
        input_layout = QtWidgets.QHBoxLayout()

        self.input_path_edit = QtWidgets.QLineEdit()
        self.input_path_edit.setPlaceholderText("选择要放大的图片或文件夹...")
        input_layout.addWidget(self.input_path_edit)

        self.btn_select_file = QtWidgets.QPushButton("选择文件")
        self.btn_select_file.clicked.connect(self.select_input_file)
        input_layout.addWidget(self.btn_select_file)

        self.btn_select_dir = QtWidgets.QPushButton("选择文件夹")
        self.btn_select_dir.clicked.connect(self.select_input_dir)
        input_layout.addWidget(self.btn_select_dir)

        input_group.setLayout(input_layout)
        layout.addWidget(input_group)

        # === 输出文件选择 ===
        output_group = QtWidgets.QGroupBox("输出路径")
        output_layout = QtWidgets.QHBoxLayout()

        self.output_path_edit = QtWidgets.QLineEdit()
        self.output_path_edit.setPlaceholderText("选择输出位置...")
        output_layout.addWidget(self.output_path_edit)

        self.btn_select_output = QtWidgets.QPushButton("选择输出")
        self.btn_select_output.clicked.connect(self.select_output)
        output_layout.addWidget(self.btn_select_output)

        output_group.setLayout(output_layout)
        layout.addWidget(output_group)

        # === 参数设置 ===
        params_group = QtWidgets.QGroupBox("放大参数")
        params_layout = QtWidgets.QGridLayout()

        # Scale (放大倍数)
        params_layout.addWidget(QtWidgets.QLabel("放大倍数:"), 0, 0)
        self.scale_combo = QtWidgets.QComboBox()
        self.scale_combo.addItems(["2", "3", "4"])
        self.scale_combo.setCurrentText("4")
        params_layout.addWidget(self.scale_combo, 0, 1)

        # Model (模型)
        params_layout.addWidget(QtWidgets.QLabel("模型:"), 0, 2)
        self.model_combo = QtWidgets.QComboBox()
        self.model_combo.addItems(
            [
                "realesr-animevideov3",
                "realesrgan-x4plus",
                "realesrgan-x4plus-anime",
            ]
        )
        self.model_combo.setCurrentText("realesrgan-x4plus-anime")
        params_layout.addWidget(self.model_combo, 0, 3)

        # Tile Size
        params_layout.addWidget(QtWidgets.QLabel("分块大小:"), 1, 0)
        self.tile_combo = QtWidgets.QComboBox()
        self.tile_combo.addItems(["0 (自动)", "32", "64", "128", "256", "512"])
        self.tile_combo.setCurrentText("0 (自动)")
        params_layout.addWidget(self.tile_combo, 1, 1)

        # GPU ID
        params_layout.addWidget(QtWidgets.QLabel("GPU:"), 1, 2)
        self.gpu_combo = QtWidgets.QComboBox()
        self.gpu_combo.addItems(["auto", "0", "1", "2"])
        self.gpu_combo.setCurrentText("auto")
        params_layout.addWidget(self.gpu_combo, 1, 3)

        # Output Format
        params_layout.addWidget(QtWidgets.QLabel("输出格式:"), 2, 0)
        self.format_combo = QtWidgets.QComboBox()
        self.format_combo.addItems(["png", "jpg", "webp"])
        self.format_combo.setCurrentText("jpg")
        params_layout.addWidget(self.format_combo, 2, 1)

        # TTA Mode
        self.tta_checkbox = QtWidgets.QCheckBox("启用 TTA 模式 (更慢但质量更好)")
        params_layout.addWidget(self.tta_checkbox, 2, 2, 1, 2)

        params_group.setLayout(params_layout)
        layout.addWidget(params_group)

        # === 控制按钮 ===
        btn_layout = QtWidgets.QHBoxLayout()

        self.btn_start = QtWidgets.QPushButton("🚀 开始放大")
        self.btn_start.setStyleSheet(
            "QPushButton { padding: 10px; font-size: 14px; font-weight: bold; }"
        )
        self.btn_start.clicked.connect(self.start_upscale)
        btn_layout.addWidget(self.btn_start)

        self.btn_clear = QtWidgets.QPushButton("清空")
        self.btn_clear.clicked.connect(self.clear_log)
        btn_layout.addWidget(self.btn_clear)

        layout.addLayout(btn_layout)

        # === 日志输出 ===
        log_group = QtWidgets.QGroupBox("处理日志")
        log_layout = QtWidgets.QVBoxLayout()

        self.log_text = QtWidgets.QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(200)
        log_layout.addWidget(self.log_text)

        log_group.setLayout(log_layout)
        layout.addWidget(log_group)

        self.setLayout(layout)

    def select_input_file(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "选择输入图片",
            str(Path.home()),
            "Images (*.png *.jpg *.jpeg *.webp);;All Files (*)",
        )
        if file_path:
            self.input_path_edit.setText(file_path)
            # 自动设置输出路径
            if not self.output_path_edit.text():
                input_file = Path(file_path)
                output_file = (
                    input_file.parent / f"{input_file.stem}_upscaled{input_file.suffix}"
                )
                self.output_path_edit.setText(str(output_file))

    def select_input_dir(self):
        dir_path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "选择输入文件夹", str(Path.home())
        )
        if dir_path:
            self.input_path_edit.setText(dir_path)
            # 自动设置输出路径
            if not self.output_path_edit.text():
                output_dir = Path(dir_path).parent / f"{Path(dir_path).name}_upscaled"
                self.output_path_edit.setText(str(output_dir))

    def select_output(self):
        input_path = self.input_path_edit.text()
        if not input_path:
            QtWidgets.QMessageBox.warning(self, "警告", "请先选择输入文件或文件夹")
            return

        if Path(input_path).is_file():
            file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self,
                "选择输出文件",
                str(Path.home()),
                "Images (*.png *.jpg *.jpeg *.webp);;All Files (*)",
            )
            if file_path:
                self.output_path_edit.setText(file_path)
        else:
            dir_path = QtWidgets.QFileDialog.getExistingDirectory(
                self, "选择输出文件夹", str(Path.home())
            )
            if dir_path:
                self.output_path_edit.setText(dir_path)

    def log(self, message: str):
        self.log_text.append(message)
        # 自动滚动到底部
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def clear_log(self):
        self.log_text.clear()

    def start_upscale(self):
        input_path = self.input_path_edit.text()
        output_path = self.output_path_edit.text()

        if not input_path:
            QtWidgets.QMessageBox.warning(self, "警告", "请选择输入文件或文件夹")
            return

        if not output_path:
            QtWidgets.QMessageBox.warning(self, "警告", "请选择输出路径")
            return

        if not Path(input_path).exists():
            QtWidgets.QMessageBox.warning(self, "警告", "输入路径不存在")
            return

        # 禁用开始按钮
        self.btn_start.setEnabled(False)
        self.btn_start.setText("处理中...")

        # 获取参数
        scale = int(self.scale_combo.currentText())
        model = self.model_combo.currentText()
        tile_size = self.tile_combo.currentText().split()[0]
        gpu_id = self.gpu_combo.currentText()
        tta = self.tta_checkbox.isChecked()
        format = self.format_combo.currentText()

        self.log(f"📂 输入: {input_path}")
        self.log(f"📂 输出: {output_path}")
        self.log(
            f"⚙️  参数: 倍数={scale}, 模型={model}, 分块={tile_size}, GPU={gpu_id}, TTA={tta}, 格式={format}"
        )
        self.log("=" * 50)

        # 创建并启动工作线程
        self.worker = UpscaleWorker(
            input_path, output_path, scale, model, tile_size, gpu_id, tta, format
        )
        self.worker.progress.connect(self.log)
        self.worker.finished.connect(self.on_finished)
        self.worker.start()

    def on_finished(self, success: bool, message: str):
        self.log("=" * 50)
        self.log(message)

        # 恢复按钮
        self.btn_start.setEnabled(True)
        self.btn_start.setText("🚀 开始放大")

        if success:
            QtWidgets.QMessageBox.information(self, "完成", "图像放大完成！")
        else:
            QtWidgets.QMessageBox.critical(
                self, "错误", "处理过程中出现错误，请查看日志"
            )


def main():
    app = QtWidgets.QApplication(sys.argv)

    # 设置应用样式
    app.setStyle("Fusion")

    # 深色主题（可选）
    palette = QtGui.QPalette()
    palette.setColor(QtGui.QPalette.ColorRole.Window, QtGui.QColor(53, 53, 53))
    palette.setColor(QtGui.QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
    palette.setColor(QtGui.QPalette.ColorRole.Base, QtGui.QColor(35, 35, 35))
    palette.setColor(QtGui.QPalette.ColorRole.AlternateBase, QtGui.QColor(53, 53, 53))
    palette.setColor(QtGui.QPalette.ColorRole.ToolTipBase, Qt.GlobalColor.white)
    palette.setColor(QtGui.QPalette.ColorRole.ToolTipText, Qt.GlobalColor.white)
    palette.setColor(QtGui.QPalette.ColorRole.Text, Qt.GlobalColor.white)
    palette.setColor(QtGui.QPalette.ColorRole.Button, QtGui.QColor(53, 53, 53))
    palette.setColor(QtGui.QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
    palette.setColor(QtGui.QPalette.ColorRole.Link, QtGui.QColor(42, 130, 218))
    palette.setColor(QtGui.QPalette.ColorRole.Highlight, QtGui.QColor(42, 130, 218))
    palette.setColor(QtGui.QPalette.ColorRole.HighlightedText, Qt.GlobalColor.black)
    app.setPalette(palette)

    window = RealESRGANGui()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
"""
realesrgan-ncnn-vulkan -i 输入图片 -o 输出图片 -n realesrgan-x4plus-anime -j 4:4:1

"""
