#!/usr/bin/python3

from PyQt5.QtWidgets import QApplication,  QDesktopWidget
from UI.MainWindow import MainWindow

def main():
    import sys
    app = QApplication(sys.argv)
    wnd = MainWindow()
    
    # ---------------------- Init Mainwindow ------------------------------
    qr = wnd.frameGeometry()
    cp = QDesktopWidget().availableGeometry().center()
    qr.moveCenter(cp)
    wnd.move(qr.topLeft())
    
    wnd.listLib.setSortingEnabled(True)

    wnd.initFaceCompare()    
    wnd.initCamera()
    # --------------------------------------------------------------------------------
    
    wnd.show()
    sys.exit(app.exec_())
    
if __name__ == '__main__':
    main()
