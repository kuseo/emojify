import sys
import string
from PyQt5.QtWidgets import *
from PyQt5 import uic

form_class = uic.loadUiType("./UI/main_window.ui")[0]

lowercase = string.ascii_lowercase
lowercase += ' '
uppercase = string.ascii_uppercase

class MyWindow(QMainWindow, form_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.pushButton.clicked.connect(self.btn_clicked)

    def btn_clicked(self):
        original_input = self.textEdit.toPlainText()
        filtered_input = []

        for v in original_input:
            if (v not in lowercase) and (v not in uppercase):
                continue
            filtered_input += v
        filtered_input = ''.join(filtered_input)
        
        #self.textBrowser_1.setText(''.join(filtered_input))
        return

if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWindow = MyWindow()
    myWindow.show()
    app.exec_()
    
    # block all elements
    # load emojify model

    # unblock elements