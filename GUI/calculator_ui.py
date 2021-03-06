# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'calculator.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Graph_Calculator(object):
    def setupUi(self, Graph_Calculator):
        Graph_Calculator.setObjectName("Graph_Calculator")
        Graph_Calculator.resize(1062, 691)
        self.lb_window = QtWidgets.QLabel(Graph_Calculator)
        self.lb_window.setGeometry(QtCore.QRect(20, 20, 421, 41))
        font = QtGui.QFont()
        font.setFamily("Euphemia")
        font.setPointSize(22)
        self.lb_window.setFont(font)
        self.lb_window.setObjectName("lb_window")
        self.lb_calc = QtWidgets.QLabel(Graph_Calculator)
        self.lb_calc.setGeometry(QtCore.QRect(40, 80, 991, 101))
        self.lb_calc.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.lb_calc.setText("")
        self.lb_calc.setObjectName("lb_calc")
        self.grid_operators = QtWidgets.QFrame(Graph_Calculator)
        self.grid_operators.setGeometry(QtCore.QRect(40, 270, 261, 251))
        self.grid_operators.setObjectName("grid_operators")
        self.gridLayout = QtWidgets.QGridLayout(self.grid_operators)
        self.gridLayout.setObjectName("gridLayout")
        self.rotate_i = QtWidgets.QSpinBox(self.grid_operators)
        self.rotate_i.setMinimum(1)
        self.rotate_i.setProperty("value", 4)
        self.rotate_i.setObjectName("rotate_i")
        self.gridLayout.addWidget(self.rotate_i, 4, 1, 1, 1)
        self.input_rotate_angle = QtWidgets.QLineEdit(self.grid_operators)
        self.input_rotate_angle.setObjectName("input_rotate_angle")
        self.gridLayout.addWidget(self.input_rotate_angle, 3, 1, 1, 1)
        self.input_trim_dist = QtWidgets.QLineEdit(self.grid_operators)
        self.input_trim_dist.setObjectName("input_trim_dist")
        self.gridLayout.addWidget(self.input_trim_dist, 2, 1, 1, 1)
        self.rotate_k = QtWidgets.QSpinBox(self.grid_operators)
        self.rotate_k.setProperty("value", 1)
        self.rotate_k.setObjectName("rotate_k")
        self.gridLayout.addWidget(self.rotate_k, 4, 2, 1, 1)
        self.btn_rotate_sp = QtWidgets.QPushButton(self.grid_operators)
        self.btn_rotate_sp.setObjectName("btn_rotate_sp")
        self.gridLayout.addWidget(self.btn_rotate_sp, 4, 0, 1, 1)
        self.btn_trim = QtWidgets.QPushButton(self.grid_operators)
        self.btn_trim.setObjectName("btn_trim")
        self.gridLayout.addWidget(self.btn_trim, 2, 0, 1, 1)
        self.btn_minkowski = QtWidgets.QPushButton(self.grid_operators)
        self.btn_minkowski.setCheckable(False)
        self.btn_minkowski.setObjectName("btn_minkowski")
        self.buttonGroup = QtWidgets.QButtonGroup(Graph_Calculator)
        self.buttonGroup.setObjectName("buttonGroup")
        self.buttonGroup.addButton(self.btn_minkowski)
        self.gridLayout.addWidget(self.btn_minkowski, 1, 0, 1, 1)
        self.btn_rotate = QtWidgets.QPushButton(self.grid_operators)
        self.btn_rotate.setObjectName("btn_rotate")
        self.gridLayout.addWidget(self.btn_rotate, 3, 0, 1, 1)
        self.btn_union = QtWidgets.QPushButton(self.grid_operators)
        self.btn_union.setCheckable(False)
        self.btn_union.setChecked(False)
        self.btn_union.setObjectName("btn_union")
        self.buttonGroup.addButton(self.btn_union)
        self.gridLayout.addWidget(self.btn_union, 0, 0, 1, 1)
        self.btn_undo = QtWidgets.QPushButton(Graph_Calculator)
        self.btn_undo.setGeometry(QtCore.QRect(420, 600, 121, 41))
        self.btn_undo.setObjectName("btn_undo")
        self.graphs = QtWidgets.QFrame(Graph_Calculator)
        self.graphs.setGeometry(QtCore.QRect(390, 260, 641, 271))
        self.graphs.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.graphs.setObjectName("graphs")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.graphs)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.btn_V151 = QtWidgets.QPushButton(self.graphs)
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.btn_V151.setFont(font)
        self.btn_V151.setObjectName("btn_V151")
        self.gridLayout_2.addWidget(self.btn_V151, 3, 0, 1, 1)
        self.btn_V1939 = QtWidgets.QPushButton(self.graphs)
        self.btn_V1939.setObjectName("btn_V1939")
        self.gridLayout_2.addWidget(self.btn_V1939, 3, 1, 1, 1)
        self.btn_S199 = QtWidgets.QPushButton(self.graphs)
        self.btn_S199.setObjectName("btn_S199")
        self.gridLayout_2.addWidget(self.btn_S199, 3, 2, 1, 1)
        self.btn_G610 = QtWidgets.QPushButton(self.graphs)
        self.btn_G610.setObjectName("btn_G610")
        self.gridLayout_2.addWidget(self.btn_G610, 3, 3, 1, 1)
        self.btn_pentagon = QtWidgets.QPushButton(self.graphs)
        self.btn_pentagon.setObjectName("btn_pentagon")
        self.gridLayout_2.addWidget(self.btn_pentagon, 4, 1, 1, 1)
        self.btn_J = QtWidgets.QPushButton(self.graphs)
        self.btn_J.setObjectName("btn_J")
        self.gridLayout_2.addWidget(self.btn_J, 0, 1, 1, 1)
        self.btn_pentagram = QtWidgets.QPushButton(self.graphs)
        self.btn_pentagram.setObjectName("btn_pentagram")
        self.gridLayout_2.addWidget(self.btn_pentagram, 4, 2, 1, 1)
        self.btn_petersen = QtWidgets.QPushButton(self.graphs)
        self.btn_petersen.setObjectName("btn_petersen")
        self.gridLayout_2.addWidget(self.btn_petersen, 4, 3, 1, 1)
        self.btn_H = QtWidgets.QPushButton(self.graphs)
        self.btn_H.setObjectName("btn_H")
        self.gridLayout_2.addWidget(self.btn_H, 0, 0, 1, 1)
        self.btn_moser = QtWidgets.QPushButton(self.graphs)
        self.btn_moser.setObjectName("btn_moser")
        self.gridLayout_2.addWidget(self.btn_moser, 4, 0, 1, 1)
        self.btn_K = QtWidgets.QPushButton(self.graphs)
        self.btn_K.setObjectName("btn_K")
        self.gridLayout_2.addWidget(self.btn_K, 0, 2, 1, 1)
        self.btn_L = QtWidgets.QPushButton(self.graphs)
        self.btn_L.setObjectName("btn_L")
        self.gridLayout_2.addWidget(self.btn_L, 0, 3, 1, 1)
        self.btn_G553 = QtWidgets.QPushButton(self.graphs)
        self.btn_G553.setObjectName("btn_G553")
        self.gridLayout_2.addWidget(self.btn_G553, 3, 4, 1, 1)
        self.btn_T = QtWidgets.QPushButton(self.graphs)
        self.btn_T.setObjectName("btn_T")
        self.gridLayout_2.addWidget(self.btn_T, 0, 4, 1, 1)
        self.btn_U = QtWidgets.QPushButton(self.graphs)
        self.btn_U.setObjectName("btn_U")
        self.gridLayout_2.addWidget(self.btn_U, 1, 0, 1, 1)
        self.btn_V = QtWidgets.QPushButton(self.graphs)
        self.btn_V.setObjectName("btn_V")
        self.gridLayout_2.addWidget(self.btn_V, 1, 1, 1, 1)
        self.btn_W = QtWidgets.QPushButton(self.graphs)
        self.btn_W.setObjectName("btn_W")
        self.gridLayout_2.addWidget(self.btn_W, 1, 2, 1, 1)
        self.btn_M = QtWidgets.QPushButton(self.graphs)
        self.btn_M.setObjectName("btn_M")
        self.gridLayout_2.addWidget(self.btn_M, 1, 3, 1, 1)
        self.btn_G = QtWidgets.QPushButton(self.graphs)
        self.btn_G.setObjectName("btn_G")
        self.gridLayout_2.addWidget(self.btn_G, 1, 4, 1, 1)
        self.horizontalLayoutWidget = QtWidgets.QWidget(Graph_Calculator)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(870, 610, 161, 51))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.btn_ok = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.btn_ok.setObjectName("btn_ok")
        self.horizontalLayout.addWidget(self.btn_ok)
        self.btn_cancel = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.btn_cancel.setObjectName("btn_cancel")
        self.horizontalLayout.addWidget(self.btn_cancel)
        self.lb_mem = QtWidgets.QLabel(Graph_Calculator)
        self.lb_mem.setGeometry(QtCore.QRect(40, 190, 991, 51))
        self.lb_mem.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.lb_mem.setText("")
        self.lb_mem.setObjectName("lb_mem")
        self.btn_mem = QtWidgets.QPushButton(Graph_Calculator)
        self.btn_mem.setGeometry(QtCore.QRect(290, 600, 121, 41))
        self.btn_mem.setObjectName("btn_mem")
        self.btn_load = QtWidgets.QPushButton(Graph_Calculator)
        self.btn_load.setGeometry(QtCore.QRect(620, 550, 171, 41))
        self.btn_load.setObjectName("btn_load")
        self.btn_merge = QtWidgets.QPushButton(Graph_Calculator)
        self.btn_merge.setGeometry(QtCore.QRect(290, 600, 121, 41))
        self.btn_merge.setObjectName("btn_merge")

        self.retranslateUi(Graph_Calculator)
        QtCore.QMetaObject.connectSlotsByName(Graph_Calculator)

    def retranslateUi(self, Graph_Calculator):
        _translate = QtCore.QCoreApplication.translate
        Graph_Calculator.setWindowTitle(_translate("Graph_Calculator", "Graph Calculator"))
        self.lb_window.setText(_translate("Graph_Calculator", "Unit Distance Graph Calculator"))
        self.input_rotate_angle.setText(_translate("Graph_Calculator", "pi"))
        self.input_trim_dist.setText(_translate("Graph_Calculator", "sqrt(3)"))
        self.btn_rotate_sp.setText(_translate("Graph_Calculator", "θ"))
        self.btn_trim.setText(_translate("Graph_Calculator", "Trim"))
        self.btn_minkowski.setText(_translate("Graph_Calculator", "Minkowski Sum"))
        self.btn_rotate.setText(_translate("Graph_Calculator", "Rotate"))
        self.btn_union.setText(_translate("Graph_Calculator", "Union"))
        self.btn_undo.setText(_translate("Graph_Calculator", "Undo"))
        self.btn_V151.setText(_translate("Graph_Calculator", "V151"))
        self.btn_V1939.setText(_translate("Graph_Calculator", "V1939"))
        self.btn_S199.setText(_translate("Graph_Calculator", "S199"))
        self.btn_G610.setText(_translate("Graph_Calculator", "G610"))
        self.btn_pentagon.setText(_translate("Graph_Calculator", "Pentagon"))
        self.btn_J.setText(_translate("Graph_Calculator", "J"))
        self.btn_pentagram.setText(_translate("Graph_Calculator", "Pentagram"))
        self.btn_petersen.setText(_translate("Graph_Calculator", "Petersen Graph"))
        self.btn_H.setText(_translate("Graph_Calculator", "H"))
        self.btn_moser.setText(_translate("Graph_Calculator", "Moser\'s Spindle"))
        self.btn_K.setText(_translate("Graph_Calculator", "K"))
        self.btn_L.setText(_translate("Graph_Calculator", "L"))
        self.btn_G553.setText(_translate("Graph_Calculator", "G553"))
        self.btn_T.setText(_translate("Graph_Calculator", "T"))
        self.btn_U.setText(_translate("Graph_Calculator", "U"))
        self.btn_V.setText(_translate("Graph_Calculator", "V"))
        self.btn_W.setText(_translate("Graph_Calculator", "W"))
        self.btn_M.setText(_translate("Graph_Calculator", "M"))
        self.btn_G.setText(_translate("Graph_Calculator", "G"))
        self.btn_ok.setText(_translate("Graph_Calculator", "Build graph"))
        self.btn_cancel.setText(_translate("Graph_Calculator", "Cancel"))
        self.btn_mem.setText(_translate("Graph_Calculator", "Memory"))
        self.btn_load.setText(_translate("Graph_Calculator", "Load graph file"))
        self.btn_merge.setText(_translate("Graph_Calculator", "Merge"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Graph_Calculator = QtWidgets.QDialog()
    ui = Ui_Graph_Calculator()
    ui.setupUi(Graph_Calculator)
    Graph_Calculator.show()
    sys.exit(app.exec_())

