import sys
import string
import numpy as np

from keras.models import Model
from keras.models import model_from_json

from emo_utils import *
from PyQt5 import QtGui
from PyQt5.QtCore import QObject, pyqtSignal, QEventLoop
from PyQt5.QtWidgets import *
from PyQt5 import uic


def sentences_to_indices(X, word_to_index, max_len):
    m = X.shape[0] # number of training examples
    
    X_indices = np.zeros((m, max_len))
    
    for i in range(m): # loop over training examples
        sentence_words = X[i].lower().split() # make input sentece as lower case and tokenize
        
        j = 0
        for w in sentence_words:
            X_indices[i, j] = word_to_index[w]
            j = j + 1
    
    return X_indices

def setElements(Window, state):
    Window.textEdit.setDisabled(state)
    Window.textBrowser_1.setDisabled(state)
    Window.textBrowser_2.setDisabled(state)
    Window.pushButton_1.setDisabled(state)
    Window.pushButton_2.setDisabled(state)


form_class = uic.loadUiType("./UI/main_window.ui")[0]

class StdoutRedirect(QObject):
    printOccur = pyqtSignal(str, str, name="print")
 
    def __init__(self, *param):
        QObject.__init__(self, None)
        self.daemon = True
        self.sysstdout = sys.stdout.write
        self.sysstderr = sys.stderr.write
 
    def stop(self):
        sys.stdout.write = self.sysstdout
        sys.stderr.write = self.sysstderr
 
    def start(self):
        sys.stdout.write = self.write
        sys.stderr.write = lambda msg : self.write(msg, color="red")
 
    def write(self, s, color="black"):
        sys.stdout.flush()
        self.printOccur.emit(s, color)

class MyWindow(QMainWindow, form_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.pushButton_1.clicked.connect(self.btn_1_load_model_clicked)
        self.pushButton_2.clicked.connect(self.btn_2_emojify_clicked)
        self.actionAbout.triggered.connect(self.menu_help_about_clicked)

        self.lowercase = string.ascii_lowercase
        self.lowercase += ' '
        self.uppercase = string.ascii_uppercase

        self.model = Model()
        self.word_to_index = {}
        self.maxLen = 10
        self.loaded = False

        self._stdout = StdoutRedirect()
        self._stdout.start()
        self._stdout.printOccur.connect(lambda x : self._append_text(x))


    def _append_text(self, msg):
        self.textBrowser_2.moveCursor(QtGui.QTextCursor.End)
        self.textBrowser_2.insertPlainText(msg)
        # refresh textedit show, refer) https://doc.qt.io/qt-5/qeventloop.html#ProcessEventsFlag-enum
        QApplication.processEvents(QEventLoop.ExcludeUserInputEvents)


    def setElements(self, state):
        self.textEdit.setDisabled(state)
        self.textBrowser_1.setDisabled(state)
        self.textBrowser_2.setDisabled(state)
        self.pushButton_1.setDisabled(state)
        self.pushButton_2.setDisabled(state)


    def btn_1_load_model_clicked(self):
        # block all elements
        self.setElements(True)

        # load emojify model
        with open("./trained model/model.json", "r") as json_file:
            model_json = json_file.read()

        self.model = model_from_json(model_json)
        self.model.load_weights("./trained model/model.h5", "r")
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        self.word_to_index, _, _ = read_glove_vecs('../data/glove.6B.50d.txt')
        

        # unblock elements
        self.setElements(False)
        self.loaded = True


    def btn_2_emojify_clicked(self):
        if self.loaded == False:
            msgbox = QMessageBox(self)
            msgbox.critical(self, "Critical", "You need to load the model first.", QMessageBox.Yes)
            return
        
        original_input = self.textEdit.toPlainText()
        filtered_input = []

        for v in original_input:
            if (v not in self.lowercase) and (v not in self.uppercase):
                continue
            filtered_input += v
        filtered_input = ''.join(filtered_input)
        
        x_test = np.array([filtered_input])
        try:
            X_test_indices = sentences_to_indices(x_test, self.word_to_index, self.maxLen)
        except KeyError as k:
            print(type(k).__name__)
            print("Unknown word : ", k, "\n")
            return
        except IndexError as i:
            print(type(i).__name__)
            print("Maximum length of input sentence is 10", "\n")
            return
        except Exception as e:
            print(type(e).__name__, ' : ', e, "\n")
            return

        self.textBrowser_1.setPlainText(original_input +' '+  label_to_emoji(np.argmax(self.model.predict(X_test_indices))))
        return


    def menu_help_about_clicked(self):
        msgbox = QMessageBox(self)
        msgbox.information(self, "About", "<a href='https://github.com/tjrkddnr/emojify'>https://github.com/tjrkddnr/emojify</a>", QMessageBox.Yes)
        return

if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWindow = MyWindow()
    myWindow.show()
    app.exec_()

    