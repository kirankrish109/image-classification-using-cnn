# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 10:36:13 2019

@author: LXI-294-VINU
"""

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QObject, pyqtSlot
from MainWindowV13 import Ui_MainWindow
import sys
from Model import Model



        

class MainWindowUIClass( Ui_MainWindow ):
    def __init__( self ):
        '''Initialize the super class
        '''
        super().__init__()
        self.model = Model()
        self.debug = True
        
    def setupUi( self, MW ):
        ''' Setup the UI of the super class, and add here code
        that relates to the way we want our UI to operate.
        '''
        super().setupUi( MW )
        
        # close the lower part of the splitter to hide the 
        # debug window under normal operations
        ##self.splitter.setSizes([300, 0])
        
        
    
        
    def runCifarTest(self):
        self.model.runCifarTest()
        self.displayMessage("CIFAR Test Ran Successfully "+self.model.testOutput)
        
    def runDisplayStatistics(self):
        self.model.displayStats(self.textBrowser, self.spinBoxBatch.value(), self.spinBoxSample.value(), self.sPlot)
        self.displayMessage("DisplayStatistics Ran Successfully")
       
        
    def runDataTests(self):
        self.model.runDataTests()
        self.displayMessage("DataTests Ran Successfully")
         
    def runImageInputTests(self):
        self.model.runImageInputTests()
        self.displayMessage("ImageInputTests Ran Successfully")
    def runKeepProbTests(self):
        self.model.runKeepProbTests()
        self.displayMessage("KeepProbTests Ran Successfully")
    def runLabelInputTests(self):
        self.model.runLabelInputTests()
        self.displayMessage("LabelInputTests Ran Successfully")
    def runNormalisationTests(self):
        self.model.runNormalisationTests()
        self.displayMessage("NormalisationTests Ran Successfully")
    def runOneHotEncodeTests(self):
        self.model.runOneHotEncodeTests()
        self.displayMessage("OneHotEncodeTests Ran Successfully")
    def runFullyConvLayerTests(self):
        self.model.runFullyConvLayerTests()
        self.displayMessage("FullyConvLayerTests Ran Successfully")
    def runConvMaxLayerTest(self):
        self.model.runConvMaxLayerTest()
        self.displayMessage("ConvMaxLayerTest Ran Successfully")
    def runOutputLayerTests(self):
        self.model.runOutputLayerTests()
        self.displayMessage("OutputLayerTests Ran Successfully")
        
    def runFlattenLayerTests(self):
        self.model.runFlattenLayerTests()
        self.displayMessage("FlattenLayerTests Ran Successfully")
        
        
    def runCNNNetworkTests(self):
        self.model.runCNNNetworkTests()
        self.displayMessage("CNNNetworkTests Ran Successfully")
    def runTrainingTests(self):
        self.model.runTrainingTests()
        self.displayMessage("TrainingTests Ran Successfully")
    
    def runPreProcessAndSave(self):
        self.model.runPreProcessAndSave()
        self.displayMessage("PreProcessAndSave Ran Successfully")
    def runShowStats(self):
        self.model.runShowStats()
        self.displayMessage("ShowStats Ran Successfully")
    def runTrainOnSingleBatch(self):
        self.model.runTrainOnSingleBatch()
        self.displayMessage("TrainOnSingleBatch Ran Successfully")
    def runFullyTrainModel(self):
        self.model.runFullyTrainModel(int(self.lineEditEpoch.text()), int(self.lineEditBatchSize.text()),float( self.lineEditKeepProb.text()))
        self.displayMessage("FullyTrainModel Ran Successfully")
        
    def runClassificationOnTestData(self):
        self.model.runClassificationOnTestData(self.cPlot)
        self.displayMessage("Classification on Test Data Completed Successfully")
       
        
    def refreshAll(self ):
        self.lineEditFolder.setText( self.model.get_cifar10_dataset_folder_path())
        self.lineEditTar.setText(self.model.get_floyd_cifar10_location())
        self.lineEditURL.setText(self.model.get_url())
        
    def debugPrint( self, msg ):
        '''Print the message in the text edit at the bottom of the
        horizontal splitter.
        '''
        self.splitter.setSizes([300, 300])
        self.textBrowserInfo.append( msg )

    # slot
    def returnPressedFolderSlot( self ):
        ''' Called when the user enters a string in the line edit and
        presses the ENTER key.
        '''
        self.debugPrint( "RETURN key pressed in LineEditFolder widget" )
        folderName =  self.lineEditFolder.text()
        if self.model.isValidFolder( folderName ):
            self.model.setFolderName( self.lineEditFolder.text() )
            self.debugPrint( "CIFAR Folder Identified" )
            self.refreshAll()
        else:
            m = QtWidgets.QMessageBox()
            m.setText("Invalid folder!\n" + folderName )
            m.setIcon(QtWidgets.QMessageBox.Warning)
            m.setStandardButtons(QtWidgets.QMessageBox.Ok
                                 | QtWidgets.QMessageBox.Cancel)
            m.setDefaultButton(QtWidgets.QMessageBox.Cancel)
            ret = m.exec_()
            self.lineEditFolder.setText( "" )
            self.debugPrint( "Invalid folder specified: " + folderName  )
            
    def returnPressedTarSlot( self ):
        ''' Called when the user enters a string in the line edit and
        presses the ENTER key.
        '''
        self.debugPrint( "RETURN key pressed in LineEditTar widget" )
        tarName =  self.lineEditTar.text()
        if self.model.isValid( tarName ):
            self.model.setTarName( self.lineEditTar.text() )
            self.model.extractTar()
            self.debugPrint( "Tar Extracted" )
            self.refreshAll()
        else:
            m = QtWidgets.QMessageBox()
            m.setText("Invalid Tar File!\n" + tarName )
            m.setIcon(QtWidgets.QMessageBox.Warning)
            m.setStandardButtons(QtWidgets.QMessageBox.Ok
                                 | QtWidgets.QMessageBox.Cancel)
            m.setDefaultButton(QtWidgets.QMessageBox.Cancel)
            ret = m.exec_()
            self.lineEditTar.setText( "" )
            self.debugPrint( "Invalid Tar file specified: " + tarName  )
    
    
    
    def returnPressedURLSlot( self ):
        ''' Called when the user enters a string in the line edit and
        presses the ENTER key.
        '''
        self.debugPrint( "RETURN key pressed in LineEditURL widget" )
        URLName =  self.lineEditURL.text()
        if self.model.isValidURL( URLName ):
            self.model.URLName =  self.lineEditURL.text() 
            self.model.downloadTar(self.textBrowserInfo, self);
            self.debugPrint( "Tar Downloaded" )
            self.refreshAll()
        else:
            m = QtWidgets.QMessageBox()
            m.setText("Invalid URL!\n" + URLName )
            m.setIcon(QtWidgets.QMessageBox.Warning)
            m.setStandardButtons(QtWidgets.QMessageBox.Ok
                                 | QtWidgets.QMessageBox.Cancel)
            m.setDefaultButton(QtWidgets.QMessageBox.Cancel)
            ret = m.exec_()
            self.lineEditURL.setText( "" )
            self.debugPrint( "Invalid URL specified: " + URLName  )
        if URLName:
            self.debugPrint( "setting URL: " + URLName )
            self.model.URLName = URLName 
            self.refreshAll()
            
            
            

    # slot
    def writeDocSlot( self ):
        ''' Called when the user presses the Write-Doc button.
        '''
        self.model.writeDoc( self.textEdit.toPlainText() )
        self.debugPrint( "Write-Doc button pressed" )

    # slot
    def browseFolderSlot( self ):
        ''' Called when the user presses the Browse button
        '''
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        folderName = QtWidgets.QFileDialog.getExistingDirectory(
                        None,
                        "QFileDialog.getOpenFileName()",
                        "",
                        options=options)
        if folderName:
            self.debugPrint( "setting CIFAR Data folder name: " + folderName )
            self.model.setFolderName( folderName )
            self.refreshAll()
            
            
    def browseTarSlot( self ):
        ''' Called when the user presses the Browse button
        '''
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        tarName, _ = QtWidgets.QFileDialog.getOpenFileName(
                        None,
                        "QFileDialog.getOpenFileName()",
                        "",
                        "Tar Files (*.tar.gz)",
                        options=options)
        if tarName:
            self.debugPrint( "setting tar file name: " + tarName )
            self.model.setTarName( tarName )
            self.model.extractTar(self.textBrowserInfo)
            self.refreshAll()
    
        
#        self.model.displayStats( self.debugTextBrowser, 1, 5 )
        
 
    def displayMessage(self, text):
        m = QtWidgets.QMessageBox()
        m.setText(text)
        m.setIcon(QtWidgets.QMessageBox.Information)
        m.setStandardButtons(QtWidgets.QMessageBox.Ok
                                 | QtWidgets.QMessageBox.Cancel)
        m.setDefaultButton(QtWidgets.QMessageBox.Cancel)
        ret = m.exec_()
        



def main():
    """
    This is the MAIN ENTRY POINT of our application.  The code at the end
    of the mainwindow.py script will not be executed, since this script is now
    our main program.   We have simply copied the code from mainwindow.py here
    since it was automatically generated by '''pyuic5'''.

    """
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = MainWindowUIClass()
    ui.setupUi(MainWindow)
    ui.refreshAll()
    MainWindow.show()
    sys.exit(app.exec_())

main()