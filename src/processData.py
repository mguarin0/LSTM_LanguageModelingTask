'''
@author: Michael Guarino
desc: this file preprocesses text data before sending it into the lstm
'''

import re, os, csv
import itertools, operator
import numpy as np
from nltk.tokenize import StanfordTokenizer
from utils import DATA_DIR, SEQUENCE_LENGTH


class ProcessData:

    def __init__(self):
        '''
        desc: constructor for class
        '''
    #end

    def _readFile(self, file):
        '''
        desc: reads in a file when passed a path
        args: path to file as a string
        returns: text corpus as string
        '''
        with open(file) as data:
            txtDoc = data.read()
        #TODO remove '\'' in text string
        txtDoc = re.sub('[.!?]\s+', '<eos> <sos>', txtDoc)
        txtDoc = re.sub('[,-]', '', txtDoc)
        return txtDoc
    #end

    def _padSeq(self, txtDocTokBySeq):
        '''
        desc: given document separated by sequence tokenized will pad the sequence so
              that all sequences are the same length
        args: given list representing a document separated by sequence tokenized
        returns: list representing a document separated by sequence tokenized
                 padded to ensure that all sequences are the same length
        '''
        for i_numSeq in range(len(txtDocTokBySeq)):
            diff = SEQUENCE_LENGTH - len(txtDocTokBySeq[i_numSeq])
            if(diff == 0):
                continue
            elif(diff < 0):
                txtDocTokBySeq[i_numSeq][:-diff]
            elif(diff > 0):
                for pad in range(diff):
                    txtDocTokBySeq[i_numSeq].append('|||$$$PAD$$$|||')
        return txtDocTokBySeq
    #end

    def _buildVocLookUp(self, txtDocTokBySeqPad, runType):
        '''
        desc: given a document separated by sequence tokenized and padded return
              a dictionary lookup for each word in the given input. The vocLookup
              dictionary is written to a csv for use during testing. If testing
              build vocLookup dictionary from vocLookup csv.
        args: list representing document separated by sequence tokenized and padded
        returns: dictionary lookup for each work in the given input
        '''
        unqVoc = sorted(set(list(itertools.chain.from_iterable(txtDocTokBySeqPad))))
        unqVoc_LookUp = {v:k for k,v in enumerate(unqVoc)}
        if (runType=='training'):
            with open('vocLookup.csv', 'w') as csv_file:
                writer = csv.writer(csv_file)
                for key, value in unqVoc_LookUp.items():
                    writer.writerow([key, value])
        elif (runType=='testing'):
            with open('vocLookup.csv', 'r') as csv_file:
                reader = csv.reader(csv_file)
                unqVoc_LookUp = {rows[0]:int(rows[1]) for rows in reader}
        return unqVoc_LookUp
    #end

    def _oneHotEncode(self, txtDocTokBySeqPad, unqVoc_LookUp, runType):
        '''
        desc: one hot encodes entire document separated by sequence tokenized
              and padded
        args: given list representing a document separated by sequence tokenized
              and a look up dictionary to determine value of words
        returns: a one hot encoded 3 dimensional matrix of training data and
                 testing data, which represents the next element in the
                 sequence
        '''
        numSeqs = len(txtDocTokBySeqPad)
        #sort words by item values
        numUnqW = sorted(unqVoc_LookUp.items(), key=operator.itemgetter(1))[-1][1]
        oheTrainData = np.zeros((numSeqs, SEQUENCE_LENGTH, numUnqW+1))
        if(runType == 'training'):
            oheTrainLabel = np.zeros((numSeqs, SEQUENCE_LENGTH, numUnqW+1))

        for i_numSeq in range(numSeqs):
            for j_seqLen, tok in enumerate(txtDocTokBySeqPad[i_numSeq]):
                if(j_seqLen==(SEQUENCE_LENGTH-2)):
                    break

                assert(tok in unqVoc_LookUp), 'not accepted, word not in corpus'
                tok = unqVoc_LookUp[tok]
                oheTrainData[i_numSeq][j_seqLen][tok] = 1

                if(runType == 'training'):
                    nextTok = unqVoc_LookUp[txtDocTokBySeqPad[i_numSeq][j_seqLen + 1]]
                    oheTrainLabel[i_numSeq][j_seqLen][nextTok] = 1

        if(runType == 'training'):
            return [oheTrainData, oheTrainLabel]
        else:
            return oheTrainData
    #end

    def oneHotDecode(self, inputOHE, unqVoc_LookUp):
        '''
        desc: decodes single one hot encoded input value
        args: given one hot encoded input
        returns: the class taht the one hot encoded vector belongs to
        '''
        indexOfclass = np.argmax(inputOHE)
        decodedClass = list(unqVoc_LookUp.keys())[list(unqVoc_LookUp.values()).index(indexOfclass)]
        return decodedClass
    #end

    def dtm_builder(self, runType):
        '''
        desc: this function coordinates all activities of data processing
              and within the function tokenizes all elements of the sequence
        returns: a one hot encoded 3 dimensional matrix of training data and
                 testing data, which represents the next element in the
                 sequence
        '''
        if (runType=='training'):
            dataFiles = ['{}/{}'.format(DATA_DIR, file) for file in os.listdir(DATA_DIR) if file.endswith('.txt')]
            allTxt = '<eos>'.join([self._readFile(file) for file in dataFiles])
        elif (runType=='testing'):
            inputString = input('Enter test string: ')
            allTxt = inputString
            assert(type(allTxt)==str), 'input must be a string'

        allTxtTok = StanfordTokenizer().tokenize(allTxt)
        allTxt_allSeq = '||*||'.join(allTxtTok).split('<eos>')
        allTxt_bySeq = [seq.split('||*||') for seq in allTxt_allSeq]
        allTxt_bySeq = [list(filter(None, seq)) for seq in allTxt_bySeq]
        for seq in allTxt_bySeq: seq.append('<eos>')
        txtDocTokBySeqPad = self._padSeq(allTxt_bySeq)
        unqVoc_LookUp = self._buildVocLookUp(txtDocTokBySeqPad, runType)
        if(runType == 'training'):
            oheTrainData, oheTrainLabel = self._oneHotEncode(txtDocTokBySeqPad, unqVoc_LookUp, runType)
            return [oheTrainData, oheTrainLabel]
        else:
            oheTrainData = self._oneHotEncode(txtDocTokBySeqPad, unqVoc_LookUp, runType)
            return [oheTrainData, unqVoc_LookUp, inputString]
    #end
#end
