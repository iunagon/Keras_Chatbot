# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 06:10:20 2023

@author: hp
"""


import numpy as np
import tensorflow as tf
import re
import time
import json
from tensorflow.keras import layers, activations, models, preprocessing

# =============================================================================
# PREPROCESSING
# =============================================================================

lines=open('conversations.json').read().split('\n')

lines1=open('utterances.jsonl').read().split('\n')


#mapping the conversation id to text
id2line={}

#line=json.loads(lines1[0])

for line in lines1:
    if line == None or line == '':
        continue;
    _line=json.loads(line)  
    id2line[_line["id"]]=_line["text"]

#print(lines[0:10])

#getting the conversation ids of corresonding conversations
lines1=lines1[:1000]

conversations_ids=[]
conversationid=json.loads(lines1[0])["conversation_id"]
temparr=[]

for line in lines1:
    if line==None or line==' ':
        continue;
    _line=json.loads(line)
    if _line["conversation_id"]==conversationid:
        temparr.append(_line["id"])
    else:
        print(temparr)#
        conversations_ids.append(temparr)
        #print(conversations_ids)
        conversationid=_line["conversation_id"]
        temparr=[_line["id"]]
    #print(conversationid)


#separating into questions and answers
questions=[]
answers=[]

for conversation in conversations_ids:
    conversation.sort()
    for i in range(len(conversation)-1):
        questions.append(id2line[conversation[i]])
        answers.append(id2line[conversation[i+1]])

#First cleaning of texts
def clean_text(text):
    text=text.lower()
    text=re.sub(r"i'm","i am",text)
    text=re.sub(r"he's","he is",text)
    text=re.sub(r"she's","she is",text)
    text=re.sub(r"that's","that is",text)
    text=re.sub(r"what's","what is",text)
    text=re.sub(r"where's","where is",text)
    text=re.sub(r"\'ll","will",text)
    text=re.sub(r"\'ve","have",text)
    text=re.sub(r"\'re","are",text)
    text=re.sub(r"\'d","would",text)
    text=re.sub(r"won't","will not",text)
    text=re.sub(r"[-()?<>\"#/@{}+=~|.,$%^&*]","",text)
    return text


#cleaning questions and answers
clean_questions=[]

for question in questions:
    clean_questions.append(clean_text(question))

clean_answers=[]

for answer in answers:
    clean_answers.append(clean_text(answer))
    

#filtering out infrequent words
word2count={}

for question in clean_questions:
    for word in question.split():
        if word not in word2count:
            word2count[word]=1 
        else:
            word2count[word]+=1 

for answer in clean_answers:
    for word in answer.split():
        if word not in word2count:
            word2count[word]=1 
        else:
            word2count[word]+=1 

#Tokeinzation: mapping words in question and answer to unique integers
threshold=20
questionswords2int={}
word_number=0
for word,count in word2count.items():
    if count>threshold:
        questionswords2int[word]=word_number;
        word_number+=1
      
word_number=0
answerswords2int={}
for word,count in word2count.items():
    if count>threshold:
        answerswords2int[word]=word_number;
        word_number+=1

#adding the last tokens to the dicts
tokens=["<PAD>","<EOS>","<OUT>","<SOS>"]

for token in tokens:
    questionswords2int[token]=len(questionswords2int)+1
    answerswords2int[token]=len(answerswords2int)+1
    #word2count[token]=
    
questionswords2int['they']=questionswords2int['<PAD>']   
questionswords2int['<PAD>']=0
 
questionswords2int['they']=questionswords2int['<PAD>']   
questionswords2int['<PAD>']=0
#creating inverse dictionary of answerswords2int
answersints2word={w_i:w for w,w_i in answerswords2int.items()}

#adding <EOS> to every answer
for i in range(len(clean_answers)):
    clean_answers[i]="<SOS> "+clean_answers[i] +" <EOS>"
    
#Vectorization: Translating to questions and answers to unique integers and 
#replacing words filtered out by <OUT>
questions_into_int=[]
for question in clean_questions:
    ints=[]
    for word in question.split():
        if word not in questionswords2int:
            ints.append(questionswords2int['<OUT>'])
        else: 
            ints.append(questionswords2int[word])
    questions_into_int.append(ints)
    
answers_into_int=[]
for answer in clean_answers:
    ints=[]
    for word in answer.split():
        if word not in answerswords2int:
            ints.append(answerswords2int['<OUT>'])
        else: 
            ints.append(answerswords2int[word])
    answers_into_int.append(ints)


#Sorting ques and answers by question length
sorted_clean_questions=[]
sorted_clean_answers=[]

for length in range(1,25+1):
    for index,ques in enumerate(questions_into_int):
        if len(ques)==length:
            sorted_clean_questions.append(questions_into_int[index])
            sorted_clean_answers.append(answers_into_int[index])
            

from keras_preprocessing.sequence import pad_sequences
from tensorflow.keras import utils

tokenizer = preprocessing.text.Tokenizer()
tokenizer.fit_on_texts( clean_questions + clean_answers )
VOCAB_SIZE = len( tokenizer.word_index )+1

#VOCAB_SIZE=len(questionswords2int)
maxlen_questions = max( [len(x) for x in sorted_clean_questions] )
encoder_inp=pad_sequences(sorted_clean_questions,maxlen=maxlen_questions, padding='post')
encoder_input_data = np.array(encoder_inp)

maxlen_answers = max( [len(x) for x in sorted_clean_answers] )
decoder_inp=pad_sequences(sorted_clean_answers,maxlen=maxlen_answers, padding='post')
decoder_input_data = np.array(decoder_inp)

for i in range(len(sorted_clean_answers)) :
    sorted_clean_answers[i] = sorted_clean_answers[i][1:]
decoded_out = pad_sequences(sorted_clean_answers,maxlen=maxlen_answers, padding='post')
onehot_answers = utils.to_categorical( decoded_out , VOCAB_SIZE )
decoder_output_data = np.array( onehot_answers )
#decoder_output_data=decoded_out
# =============================================================================
# BUILDING THE NEURALNETWORK ARCHITECTURE AND TRAINING
# =============================================================================


encoder_inputs = tf.keras.layers.Input(shape=( maxlen_questions , ))
encoder_embedding = tf.keras.layers.Embedding( VOCAB_SIZE, 200 , mask_zero=True ) (encoder_inputs)
encoder_outputs , state_h , state_c = tf.keras.layers.LSTM( 200 , return_state=True )( encoder_embedding )
encoder_states = [ state_h , state_c ]


#
decoder_inputs = tf.keras.layers.Input(shape=( maxlen_answers ,  ))
decoder_embedding = tf.keras.layers.Embedding( VOCAB_SIZE, 200 , mask_zero=True) (decoder_inputs)
decoder_lstm = tf.keras.layers.LSTM( 200 , return_state=True , return_sequences=True )
decoder_outputs , _ , _ = decoder_lstm( decoder_embedding , initial_state=encoder_states )
decoder_dense = tf.keras.layers.Dense( VOCAB_SIZE , activation=tf.keras.activations.softmax ) 
output = decoder_dense ( decoder_outputs )

model = tf.keras.models.Model([encoder_inputs, decoder_inputs], output )
model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='categorical_crossentropy')

model.summary()

model.fit([encoder_input_data , decoder_input_data], decoder_output_data, batch_size=50, epochs=150 ) 
model.save( 'model1.h5' )



#DEFINING MODELS TO TEST/INFERENCE
def make_inference_models():
    
    encoder_model = tf.keras.models.Model(encoder_inputs, encoder_states)
    
    decoder_state_input_h = tf.keras.layers.Input(shape=( 200 ,))
    decoder_state_input_c = tf.keras.layers.Input(shape=( 200 ,))
    
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    
    decoder_outputs, state_h, state_c = decoder_lstm(
        decoder_embedding , initial_state=decoder_states_inputs)
    
    decoder_states = [state_h, state_c]

    decoder_outputs = decoder_dense(decoder_outputs)
    
    decoder_model = tf.keras.models.Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states)
    
    return encoder_model , decoder_model


def str_to_tokens( sentence : str ):

    words = sentence.lower().split()
    tokens_list = list()
  
    for word in words:
        tokens_list.append( tokenizer.word_index[ word ] ) 
    return preprocessing.sequence.pad_sequences( [tokens_list] , maxlen=maxlen_questions , padding='post')


enc_model , dec_model = make_inference_models()


prepro1 = ""
while prepro1 != 'q':
    
    prepro1 = input("you : ")
    try:
        prepro1 = clean_text(prepro1)
        prepro = [prepro1]
        
        txt = []
        for x in prepro:
            lst = []
            for y in x.split():
                try:
                    lst.append(tokenizer.word_index[y])
                except:
                    lst.append(tokenizer.word_index['<OUT>'])
            txt.append(lst)
        txt = pad_sequences(txt, maxlen_questions, padding='post')


        ###
        stat = enc_model.predict( txt )

        empty_target_seq = np.zeros( ( 1 , 1) )
        empty_target_seq[0, 0] = questionswords2int['<SOS>']
        stop_condition = False
        decoded_translation = ''


        while not stop_condition :

            dec_outputs , h , c = dec_model.predict([ empty_target_seq ] + stat )

           

            sampled_word_index = np.argmax( dec_outputs[0, -1, :] )

            sampled_word = answersints2word[sampled_word_index] + ' '

            if sampled_word != '<EOS> ':
                decoded_translation += sampled_word           


            if sampled_word == '<EOS> ' or len(decoded_translation.split()) > 15:
                stop_condition = True

            empty_target_seq = np.zeros( ( 1 , 1 ) )  
            empty_target_seq[ 0 , 0 ] = sampled_word_index
            stat = [ h , c ] 

        print("Chatbot : ", decoded_translation )
        print("==============================================")

    except Exception:
        print("sorry didn't understand you , please type again :( ")














