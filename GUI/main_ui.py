# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'UDG.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(663, 594)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.lb_main = QtWidgets.QLabel(self.centralwidget)
        self.lb_main.setGeometry(QtCore.QRect(40, 30, 421, 51))
        font = QtGui.QFont()
        font.setFamily("Euphemia")
        font.setPointSize(26)
        self.lb_main.setFont(font)
        self.lb_main.setObjectName("lb_main")
        self.G_options = QtWidgets.QWidget(self.centralwidget)
        self.G_options.setGeometry(QtCore.QRect(180, 260, 291, 201))
        self.G_options.setObjectName("G_options")
        self.G_layout = QtWidgets.QGridLayout(self.G_options)
        self.G_layout.setContentsMargins(0, 0, 0, 0)
        self.G_layout.setObjectName("G_layout")
        self.btn_draw = QtWidgets.QPushButton(self.G_options)
        self.btn_draw.setObjectName("btn_draw")
        self.G_layout.addWidget(self.btn_draw, 0, 0, 1, 1)
        self.randomName = QtWidgets.QCheckBox(self.G_options)
        self.randomName.setChecked(True)
        self.randomName.setObjectName("randomName")
        self.G_layout.addWidget(self.randomName, 0, 2, 1, 1)
        self.btn_color = QtWidgets.QPushButton(self.G_options)
        self.btn_color.setObjectName("btn_color")
        self.G_layout.addWidget(self.btn_color, 4, 0, 1, 1)
        self.drawScaling = QtWidgets.QDoubleSpinBox(self.G_options)
        self.drawScaling.setMaximumSize(QtCore.QSize(133, 16777215))
        self.drawScaling.setToolTipDuration(-1)
        self.drawScaling.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.drawScaling.setLocale(QtCore.QLocale(QtCore.QLocale.C, QtCore.QLocale.AnyCountry))
        self.drawScaling.setFrame(True)
        self.drawScaling.setDecimals(1)
        self.drawScaling.setMinimum(0.5)
        self.drawScaling.setMaximum(3.0)
        self.drawScaling.setSingleStep(0.25)
        self.drawScaling.setProperty("value", 1.0)
        self.drawScaling.setObjectName("drawScaling")
        self.G_layout.addWidget(self.drawScaling, 0, 1, 1, 1)
        self.saveColoring = QtWidgets.QCheckBox(self.G_options)
        self.saveColoring.setChecked(True)
        self.saveColoring.setObjectName("saveColoring")
        self.G_layout.addWidget(self.saveColoring, 5, 1, 1, 1)
        self.maxColors = QtWidgets.QSpinBox(self.G_options)
        self.maxColors.setWrapping(False)
        self.maxColors.setButtonSymbols(QtWidgets.QAbstractSpinBox.UpDownArrows)
        self.maxColors.setMinimum(1)
        self.maxColors.setMaximum(6)
        self.maxColors.setProperty("value", 4)
        self.maxColors.setObjectName("maxColors")
        self.G_layout.addWidget(self.maxColors, 4, 1, 1, 1)
        self.dimacs = QtWidgets.QCheckBox(self.G_options)
        self.dimacs.setObjectName("dimacs")
        self.G_layout.addWidget(self.dimacs, 5, 2, 1, 1)
        self.btn_print = QtWidgets.QPushButton(self.G_options)
        self.btn_print.setObjectName("btn_print")
        self.G_layout.addWidget(self.btn_print, 5, 0, 1, 1)
        self.btn_clear = QtWidgets.QPushButton(self.G_options)
        self.btn_clear.setObjectName("btn_clear")
        self.G_layout.addWidget(self.btn_clear, 6, 1, 1, 1)
        self.lb_G = QtWidgets.QLabel(self.centralwidget)
        self.lb_G.setGeometry(QtCore.QRect(40, 100, 581, 81))
        font = QtGui.QFont()
        font.setFamily("Euphemia")
        font.setPointSize(22)
        self.lb_G.setFont(font)
        self.lb_G.setFrameShape(QtWidgets.QFrame.Box)
        self.lb_G.setText("")
        self.lb_G.setObjectName("lb_G")
        self.verticalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(100, 480, 451, 101))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.graphmodifier = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.graphmodifier.setContentsMargins(0, 0, 0, 0)
        self.graphmodifier.setObjectName("graphmodifier")
        self.btn_graph = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.btn_graph.setCheckable(False)
        self.btn_graph.setObjectName("btn_graph")
        self.graphmodifier.addWidget(self.btn_graph)
        self.btn_operators = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.btn_operators.setObjectName("btn_operators")
        self.graphmodifier.addWidget(self.btn_operators)
        self.btn_load = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.btn_load.setObjectName("btn_load")
        self.graphmodifier.addWidget(self.btn_load)
        self.graphData = QtWidgets.QFrame(self.centralwidget)
        self.graphData.setGeometry(QtCore.QRect(410, 200, 206, 68))
        self.graphData.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.graphData.setObjectName("graphData")
        self.gridLayout = QtWidgets.QGridLayout(self.graphData)
        self.gridLayout.setObjectName("gridLayout")
        self.lb_m = QtWidgets.QLabel(self.graphData)
        font = QtGui.QFont()
        font.setFamily("Euphemia")
        font.setPointSize(16)
        self.lb_m.setFont(font)
        self.lb_m.setIndent(10)
        self.lb_m.setObjectName("lb_m")
        self.gridLayout.addWidget(self.lb_m, 2, 0, 1, 1)
        self.lb_n = QtWidgets.QLabel(self.graphData)
        font = QtGui.QFont()
        font.setFamily("Euphemia")
        font.setPointSize(16)
        self.lb_n.setFont(font)
        self.lb_n.setIndent(10)
        self.lb_n.setObjectName("lb_n")
        self.gridLayout.addWidget(self.lb_n, 1, 0, 1, 1)
        self.btn_M_property = QtWidgets.QPushButton(self.centralwidget)
        self.btn_M_property.setGeometry(QtCore.QRect(490, 330, 121, 51))
        self.btn_M_property.setObjectName("btn_M_property")
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Unit Distance Graphs"))
        self.lb_main.setText(_translate("MainWindow", "Unit Distance Graphs App"))
        self.btn_draw.setText(_translate("MainWindow", "Draw G"))
        self.randomName.setText(_translate("MainWindow", "Random name"))
        self.btn_color.setText(_translate("MainWindow", "Color G"))
        self.drawScaling.setToolTip(_translate("MainWindow", "Rescaling factor"))
        self.saveColoring.setText(_translate("MainWindow", "Save Coloring"))
        self.dimacs.setText(_translate("MainWindow", "DIMACS"))
        self.btn_print.setText(_translate("MainWindow", "Save Graph"))
        self.btn_clear.setText(_translate("MainWindow", "Clear graph"))
        self.btn_graph.setText(_translate("MainWindow", "Create graph G"))
        self.btn_operators.setText(_translate("MainWindow", "Operators on G"))
        self.btn_load.setText(_translate("MainWindow", "Load graph"))
        self.lb_m.setText(_translate("MainWindow", "m = 0"))
        self.lb_n.setText(_translate("MainWindow", "n = 0"))
        self.btn_M_property.setText(_translate("MainWindow", "Check property on M"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

