import os
# Force a safe GUI setup on WSL2: X11 (xcb), software path, and no stray plugin paths
os.environ.pop("QT_PLUGIN_PATH", None)
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
os.environ.setdefault("QT_OPENGL", "software")

import sys
from PyQt5 import QtWidgets, QtCore
QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_UseSoftwareOpenGL)

from .gui.main_window import MainWindow

def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
