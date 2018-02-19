'''
@author: Michael Guarino
desc: this file coordinates all activity within the project
'''
import sys
import processData as pD
from lstm import train_model, test_model, load_model

def run(runType):
    '''
    desc:
    args:
    returns:
    '''
    dataMan = pD.ProcessData()
    if(runType=='training'):
        oheTrainData, oheTrainLabel = dataMan.dtm_builder(runType)
        train_model(oheTrainData, oheTrainLabel)
    elif(runType=='testing'):
        lstm_loaded_model = load_model()
        while True:
            oheTrainData, unqVoc_LookUp, inputString = dataMan.dtm_builder(runType)
            test_model(lstm_loaded_model, oheTrainData, unqVoc_LookUp, inputString)
#end

if __name__ == "__main__":
    assert(sys.argv[1]=='training' or sys.argv[1]=='testing'), "please pass valid input"
    if(sys.argv[1]=='training'):
        runType = 'training'
    elif(sys.argv[1]=='testing'):
        runType = 'testing'
    run(runType)
#end
