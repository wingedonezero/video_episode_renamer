#!/usr/bin/env python3
"""
Video Episode Renamer - Main GUI Application
Matches and renames video files based on reference files with correct names
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QFileDialog, QTableWidget,
    QTableWidgetItem, QHeaderView, QProgressBar, QMessageBox,
    QComboBox, QSlider, QSpinBox, QGroupBox, QDialog, QDialogButtonBox
)
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtGui import QColor, QAction

from core.pipeline import MatchingPipeline
from core.cache import MediaCache
from utils.config import Config

@dataclass
class MatchResult:
    remux_path: Path
    reference_path: Optional[Path]
    confidence: float
    info: str

class MatcherThread(QThread):
    progress = pyqtSignal(str, int)
    match_found = pyqtSignal(dict)
    finished = pyqtSignal()

    def __init__(self, pipeline, references, remuxes):
        super().__init__()
        self.pipeline = pipeline
        self.references = references
        self.remuxes = remuxes
        self._stop_requested = False

    def run(self):
        try:
            for result in self.pipeline.match(self.references, self.remuxes):
                if self._stop_requested:
                    break
                if result['type'] == 'progress':
                    self.progress.emit(result['message'], result['value'])
                elif result['type'] == 'match':
                    self.match_found.emit(result['data'])
        except Exception as e:
            self.progress.emit(f"Error: {str(e)}", 0)
        finally:
            self.finished.emit()

    def stop(self):
        self._stop_requested = True
        self.pipeline.stop()

class SettingsDialog(QDialog):
    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.config = config
        self.setWindowTitle("Settings")
        self.setMinimumWidth(500)
        layout = QVBoxLayout(self)
        general_group = QGroupBox("General Matching Settings")
        general_layout = QHBoxLayout()
        general_layout.addWidget(QLabel("Analysis Start Offset (%):"))
        self.offset_spinbox = QSpinBox()
        self.offset_spinbox.setRange(0, 40)
        self.offset_spinbox.setSuffix(" %")
        self.offset_spinbox.setToolTip("Skips this percentage of the file's beginning before analysis. Affects Chromaprint, pHash, and MFCC.")
        general_layout.addWidget(self.offset_spinbox)
        general_layout.addStretch()
        general_group.setLayout(general_layout)
        layout.addWidget(general_group)
        panako_group = QGroupBox("Panako Fingerprinter")
        panako_layout = QHBoxLayout()
        panako_layout.addWidget(QLabel("panako.jar Path:"))
        self.panako_path_edit = QLineEdit()
        panako_layout.addWidget(self.panako_path_edit)
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self.browse_for_panako_jar)
        panako_layout.addWidget(browse_btn)
        panako_group.setLayout(panako_layout)
        layout.addWidget(panako_group)
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        self.load_settings()

    def browse_for_panako_jar(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Panako JAR", "", "JAR Files (*.jar)")
        if file_path:
            self.panako_path_edit.setText(file_path)

    def load_settings(self):
        settings = self.config.load()
        self.panako_path_edit.setText(settings.get('panako_jar', ''))
        self.offset_spinbox.setValue(settings.get('analysis_start_percent', 15))

    def get_settings(self) -> dict:
        return {
            'panako_jar': self.panako_path_edit.text(),
            'analysis_start_percent': self.offset_spinbox.value()
        }

class VideoEpisodeRenamer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.config = Config()
        self.cache = MediaCache()
        self.app_data_dir = Path.cwd() / "app_data"
        self.app_data_dir.mkdir(exist_ok=True)
        self.pipeline = MatchingPipeline(self.cache, self.config, self.app_data_dir)
        self.matcher_thread = None
        self.match_results = []
        self.init_ui()
        self.load_settings()

    def init_ui(self):
        self.setWindowTitle("Video Episode Renamer")
        self.setGeometry(100, 100, 1400, 800)
        self.init_menu()
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        folder_group = QGroupBox("Folder Selection")
        folder_layout = QVBoxLayout()
        ref_layout = QHBoxLayout()
        ref_layout.addWidget(QLabel("Reference (Correctly Named):"))
        self.ref_folder = QLineEdit()
        ref_layout.addWidget(self.ref_folder)
        ref_btn = QPushButton("Browse...")
        ref_btn.clicked.connect(lambda: self.select_folder(self.ref_folder))
        ref_layout.addWidget(ref_btn)
        folder_layout.addLayout(ref_layout)
        remux_layout = QHBoxLayout()
        remux_layout.addWidget(QLabel("Remux (To Rename):"))
        self.remux_folder = QLineEdit()
        remux_layout.addWidget(self.remux_folder)
        remux_btn = QPushButton("Browse...")
        remux_btn.clicked.connect(lambda: self.select_folder(self.remux_folder))
        remux_layout.addWidget(remux_btn)
        folder_layout.addLayout(remux_layout)
        folder_group.setLayout(folder_layout)
        layout.addWidget(folder_group)
        config_group = QGroupBox("Matching Configuration")
        config_layout = QHBoxLayout()
        config_layout.addWidget(QLabel("Mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Correlation (Audio)", "Chromaprint (Audio)", "MFCC (Audio)", "Panako (Audio)", "Perceptual Hash (Video)", "Scene Detection (Video)"])
        self.mode_combo.currentTextChanged.connect(self.on_mode_changed)
        config_layout.addWidget(self.mode_combo)
        self.lang_label = QLabel("Language:")
        config_layout.addWidget(self.lang_label)
        self.lang_input = QLineEdit()
        self.lang_input.setMaximumWidth(60)
        self.lang_input.setPlaceholderText("jpn")
        config_layout.addWidget(self.lang_input)
        config_layout.addWidget(QLabel("Min Confidence:"))
        self.confidence_slider = QSlider(Qt.Orientation.Horizontal)
        self.confidence_slider.setRange(50, 95)
        self.confidence_slider.setValue(75)
        self.confidence_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.confidence_slider.setTickInterval(5)
        config_layout.addWidget(self.confidence_slider)
        self.confidence_label = QLabel("75%")
        self.confidence_slider.valueChanged.connect(lambda v: self.confidence_label.setText(f"{v}%"))
        config_layout.addWidget(self.confidence_label)
        config_layout.addStretch()
        config_group.setLayout(config_layout)
        layout.addWidget(config_group)
        control_layout = QHBoxLayout()
        self.start_btn = QPushButton("Start Matching")
        self.start_btn.clicked.connect(self.start_matching)
        control_layout.addWidget(self.start_btn)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_matching)
        self.stop_btn.setEnabled(False)
        control_layout.addWidget(self.stop_btn)
        self.clear_cache_btn = QPushButton("Clear Cache")
        self.clear_cache_btn.clicked.connect(self.clear_cache)
        control_layout.addWidget(self.clear_cache_btn)
        control_layout.addStretch()
        self.rename_btn = QPushButton("Rename Matched Files")
        self.rename_btn.clicked.connect(self.rename_files)
        self.rename_btn.setEnabled(False)
        control_layout.addWidget(self.rename_btn)
        layout.addLayout(control_layout)
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(5)
        self.results_table.setHorizontalHeaderLabels(["Original Name", "Proposed Name", "Confidence", "Match Info", "Status"])
        header = self.results_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.Fixed)
        self.results_table.setColumnWidth(2, 100)
        self.results_table.setColumnWidth(4, 100)
        layout.addWidget(self.results_table)
        self.progress = QProgressBar()
        self.statusBar().addPermanentWidget(self.progress, 1)
        self.status_label = QLabel("Ready")
        self.statusBar().addWidget(self.status_label)

    def init_menu(self):
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("&File")
        settings_action = QAction("Settings...", self)
        settings_action.triggered.connect(self.open_settings_dialog)
        file_menu.addAction(settings_action)

    def open_settings_dialog(self):
        dialog = SettingsDialog(self.config, self)
        if dialog.exec():
            new_settings = dialog.get_settings()
            current_settings = self.config.load()
            current_settings.update(new_settings)
            self.config.save(current_settings)
            self.status_label.setText("Settings saved.")

    def on_mode_changed(self, mode_text):
        is_audio = "Audio" in mode_text
        self.lang_label.setVisible(is_audio)
        self.lang_input.setVisible(is_audio)
        if "Panako" in mode_text: self.confidence_slider.setValue(80)
        elif "Video" in mode_text: self.confidence_slider.setValue(85)
        else: self.confidence_slider.setValue(75)

    def select_folder(self, line_edit):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder: line_edit.setText(folder)

    def start_matching(self):
        if not self.ref_folder.text() or not self.remux_folder.text():
            QMessageBox.warning(self, "Error", "Please select both folders"); return

        ref_files = self.get_video_files(Path(self.ref_folder.text()))
        remux_files = self.get_video_files(Path(self.remux_folder.text()))

        if not ref_files or not remux_files:
            QMessageBox.warning(self, "Error", "No video files found in one or both of the selected folders.")
            return

        # --- MODIFICATION: Provide immediate UI feedback ---
        self.status_label.setText(f"Starting {self.mode_combo.currentText()}...")
        self.progress.setValue(0)
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.rename_btn.setEnabled(False)
        # --- END MODIFICATION ---

        mode_map = {"Correlation (Audio)": "correlation", "Chromaprint (Audio)": "chromaprint", "MFCC (Audio)": "mfcc", "Panako (Audio)": "panako", "Perceptual Hash (Video)": "phash", "Scene Detection (Video)": "scene"}
        mode = mode_map[self.mode_combo.currentText()]
        self.pipeline.set_mode(mode)
        self.pipeline.set_language(self.lang_input.text() or None)
        self.pipeline.set_threshold(self.confidence_slider.value() / 100.0)

        self.results_table.setRowCount(0)
        self.match_results.clear()

        self.matcher_thread = MatcherThread(self.pipeline, ref_files, remux_files)
        self.matcher_thread.progress.connect(self.update_progress)
        self.matcher_thread.match_found.connect(self.add_match_result)
        self.matcher_thread.finished.connect(self.matching_finished)
        self.matcher_thread.start()

    def stop_matching(self):
        if self.matcher_thread:
            self.matcher_thread.stop()
            self.stop_btn.setEnabled(False)
            self.status_label.setText("Stopping...")

    def update_progress(self, message, value):
        self.status_label.setText(message)
        self.progress.setValue(value)

    def add_match_result(self, match_data):
        self.match_results.append(match_data)
        row = self.results_table.rowCount()
        self.results_table.insertRow(row)
        orig_item = QTableWidgetItem(Path(match_data['remux_path']).name)
        self.results_table.setItem(row, 0, orig_item)
        if match_data.get('reference_path'): proposed = QTableWidgetItem(Path(match_data['reference_path']).name)
        else: proposed = QTableWidgetItem("")
        self.results_table.setItem(row, 1, proposed)
        conf = match_data.get('confidence', 0)
        conf_item = QTableWidgetItem(f"{conf:.1%}")
        self.results_table.setItem(row, 2, conf_item)
        info_item = QTableWidgetItem(match_data.get('info', ''))
        self.results_table.setItem(row, 3, info_item)
        threshold = self.confidence_slider.value() / 100.0
        if match_data.get('reference_path') and conf >= threshold:
            status = "Matched"
            if conf > 0.95: color = QColor(144, 238, 144)
            else: color = QColor(255, 255, 150)
        elif match_data.get('reference_path'):
            status = "Low Confidence"; color = QColor(255, 200, 150)
        else:
            status = "Unmatched"; color = QColor(255, 182, 193)
        status_item = QTableWidgetItem(status)
        self.results_table.setItem(row, 4, status_item)
        for col in range(5):
            item = self.results_table.item(row, col)
            if item:
                item.setBackground(color)
                item.setForeground(QColor(0, 0, 0))
        if status == "Matched": self.rename_btn.setEnabled(True)

    def matching_finished(self):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("Matching complete")
        self.progress.setValue(100)

    def rename_files(self):
        threshold = self.confidence_slider.value() / 100.0
        rename_list = []
        for match in self.match_results:
            if match.get('reference_path') and match.get('confidence', 0) >= threshold:
                orig = Path(match['remux_path']); ref = Path(match['reference_path'])
                new_name = ref.stem + orig.suffix; new_path = orig.parent / new_name
                rename_list.append((orig, new_path))
        if not rename_list: QMessageBox.information(self, "Info", "No files to rename"); return
        msg = f"Rename {len(rename_list)} files?\n\nExamples:\n"
        for orig, new in rename_list[:3]: msg += f"{orig.name} → {new.name}\n"
        if len(rename_list) > 3: msg += f"... and {len(rename_list) - 3} more"
        reply = QMessageBox.question(self, "Confirm Rename", msg)
        if reply != QMessageBox.StandardButton.Yes: return
        success, errors = 0, []
        for orig, new in rename_list:
            try:
                if new.exists(): errors.append(f"{new.name} already exists"); continue
                orig.rename(new); success += 1
                for row in range(self.results_table.rowCount()):
                    if self.results_table.item(row, 0).text() == orig.name:
                        self.results_table.item(row, 0).setText(new.name)
                        self.results_table.item(row, 4).setText("Renamed"); break
            except Exception as e: errors.append(f"{orig.name}: {str(e)}")
        msg = f"Successfully renamed {success} files"
        if errors:
            msg += f"\n\nErrors:\n" + "\n".join(errors[:5])
            if len(errors) > 5: msg += f"\n... and {len(errors) - 5} more"
        QMessageBox.information(self, "Rename Complete", msg)
        self.rename_btn.setEnabled(False)

    def clear_cache(self):
        self.cache.clear(); self.status_label.setText("Cache cleared")

    def get_video_files(self, folder: Path) -> List[Path]:
        extensions = {'.mkv', '.mp4', '.avi', '.mov', '.ts', '.m2ts'}
        files = []
        for ext in extensions:
            files.extend(folder.glob(f"*{ext}")); files.extend(folder.glob(f"*{ext.upper()}"))
        return sorted(files)

    def load_settings(self):
        try:
            settings = self.config.load()
            if 'ref_folder' in settings: self.ref_folder.setText(settings['ref_folder'])
            if 'remux_folder' in settings: self.remux_folder.setText(settings['remux_folder'])
            if 'language' in settings: self.lang_input.setText(settings['language'])
            if 'mode' in settings:
                idx = self.mode_combo.findText(settings['mode'])
                if idx >= 0: self.mode_combo.setCurrentIndex(idx)
            if 'confidence' in settings: self.confidence_slider.setValue(settings['confidence'])
        except: pass

    def closeEvent(self, event):
        settings = {'ref_folder': self.ref_folder.text(), 'remux_folder': self.remux_folder.text(), 'language': self.lang_input.text(), 'mode': self.mode_combo.currentText(), 'confidence': self.confidence_slider.value()}
        all_settings = self.config.load()
        all_settings.update(settings)
        self.config.save(all_settings)
        if self.matcher_thread and self.matcher_thread.isRunning():
            self.matcher_thread.stop(); self.matcher_thread.wait()
        event.accept()

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Video Episode Renamer")
    window = VideoEpisodeRenamer()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
