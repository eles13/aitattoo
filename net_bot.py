style_transfer_dir = './Real-time-multi-style-transfer/'
import telebot
from telebot import TeleBot
from telebot import types
import os
import time
import numpy as np
import logging
#from pymorphy2 import MorphAnalyzer
import requests
import wget
import pickle
import sys
import pickle
import argparse
from os.path import join
import scipy.misc
import random
import imageio
import model
if style_transfer_dir not in sys.path:
    sys.path.append(style_transfer_dir)
import torch
from torchvision import transforms
from inference.Inferencer import Inferencer
from models.PasticheModel import PasticheModel
from PIL import Image
import tensorflow as tf
import pika
import base64
import joblib
from sklearn.linear_model import LogisticRegression
tf.compat.v1.disable_eager_execution()
from own_style import *

connection = pika.BlockingConnection(pika.ConnectionParameters(heartbeat=0 ,socket_timeout=100000))
channel = connection.channel()
channel.queue_declare(queue='toenc')

#################################################################
#############################SETUP###############################
#################################################################
pic_path = './saved_pics/'
text_rep_path = './texts/'
own_styles_path = './own_styles/'
own_pics_path = './own_pics/'

modes = {}
stylesdct = joblib.load('./stylesNames.dict')
ownstyle_transfer_dict = {}

from string import punctuation
 
def clean(text):
    if not isinstance(text, str):
        raise TypeError('text должен быть str')
    return ''.join(x for x in text.lower() if x not in punctuation)

os.makedirs(pic_path, exist_ok=True)
os.makedirs(text_rep_path, exist_ok=True)
os.makedirs(own_styles_path, exist_ok=True)
os.makedirs(own_pics_path, exist_ok=True)

logging.basicConfig(filename="botlog.log", level=logging.INFO, filemode='w')
token = '1277130373:AAH7oKdzkHwopPmwVluIIVPFfU_rugTedRM'
bot = TeleBot(token)

device = torch.device("cpu")

num_styles = 33
image_size = 512

pastichemodel = PasticheModel(num_styles)
model_save_dir = style_transfer_dir + "checkpoints/pastichemodel_1-FINAL.pth"

pastichemodel = PasticheModel(num_styles)

inference = Inferencer(pastichemodel,device,image_size)
inference.load_model_weights(model_save_dir)

flowerlog = joblib.load('./logreg.m')

logging.info('---started---')
running = False

logging.info('Started loading and setting model')
def load_training_data(data_dir, data_set, caption_vector_length, n_classes):
    if data_set == 'flowers':
        flower_str_captions = pickle.load(
            open(join(data_dir, 'flowers', 'flowers_caps.pkl'), "rb"))

        img_classes = pickle.load(
            open(join(data_dir, 'flowers', 'flower_tc.pkl'), "rb"))

        flower_enc_captions = pickle.load(
            open(join(data_dir, 'flowers', 'flower_tv.pkl'), "rb"))
        # h1 = h5py.File(join(data_dir, 'flower_tc.hdf5'))
        tr_image_ids = pickle.load(
            open(join(data_dir, 'flowers', 'train_ids.pkl'), "rb"))
        val_image_ids = pickle.load(
            open(join(data_dir, 'flowers', 'val_ids.pkl'), "rb"))

        # n_classes = n_classes
        max_caps_len = caption_vector_length

        tr_n_imgs = len(tr_image_ids)
        val_n_imgs = len(val_image_ids)

        return {
            'image_list': tr_image_ids,
            'captions': flower_enc_captions,
            'data_length': tr_n_imgs,
            'classes': img_classes,
            'n_classes': n_classes,
            'max_caps_len': max_caps_len,
            'val_img_list': val_image_ids,
            'val_captions': flower_enc_captions,
            'val_data_len': val_n_imgs,
            'str_captions': flower_str_captions,
        }

    else:
        raise Exception('This dataset has not been handeled yet. '
                         'Contributions are welcome.')


datasets_root_dir = 'datasets'

loaded_data = load_training_data(datasets_root_dir, 'flowers', 512, 312)
model_options = {
    'z_dim': 100,
    't_dim': 256,
    'batch_size': 64,
    'image_size': 128,
    'gf_dim': 64,
    'df_dim': 64,
    'caption_vector_length': 512,
    'n_classes': 312
}

gan = model.GAN(model_options)
input_tensors, variables, loss, outputs, checks = gan.build_model()

sessdefault = tf.compat.v1.InteractiveSession()
sessflowers = tf.compat.v1.InteractiveSession()
tf.compat.v1.initialize_all_variables().run()

saver = tf.compat.v1.train.Saver(max_to_keep=10000)
logging.info('Trying to resume model from ' + str(tf.train.latest_checkpoint('./checkpoints/')))
if tf.train.latest_checkpoint('./checkpoints/default/') is not None:
    saver.restore(sessdefault, tf.train.latest_checkpoint('./checkpoints/default/'))
    logging.info('Successfully loaded model from ./checkpoints/default/')
else:
    logging.error('Could not load checkpoints. Please provide a valid path to'
          ' your checkpoints directory')
    exit()
if tf.train.latest_checkpoint('./checkpoints/flowers/') is not None:
    saver.restore(sessflowers, tf.train.latest_checkpoint('./checkpoints/flowers/'))
    logging.info('Successfully loaded model from ./checkpoints/flowers/')
else:
    logging.error('Could not load checkpoints. Please provide a valid path to'
          ' your checkpoints directory')
    exit()
#################################################################

@bot.message_handler(commands = ['start'])
def startBot(message):
    global running
    logging.info('Preparing to reply to "start" command ' + str(message.chat.id))
    bot.reply_to(message,'Hi, this is an ai tattoo bot! Send me a description of your desirable picture and there will be magic')
    logging.info('Replied to "start" command ' + str(message.chat.id))
    running = True
    return

@bot.message_handler(commands = ['mode'])
def changeMode(message):
    if message.chat.id not in modes.keys():
        modes[message.chat.id] = 0
    logging.info('Received "mode" command ' + str(message.chat.id))
    keyboard = types.InlineKeyboardMarkup(row_width = 1)
    keyboard.row(types.InlineKeyboardButton(text='generate and stylize', callback_data='mode|0'))
    keyboard.row(types.InlineKeyboardButton(text='only generate', callback_data='mode|1'))
    keyboard.row(types.InlineKeyboardButton(text='retrieve from web and stylize', callback_data='mode|2'))
    keyboard.row(types.InlineKeyboardButton(text='stylize my image', callback_data='mode|3'))
    bot.send_message(message.chat.id, text = 'Choose mode', reply_markup=keyboard)
    logging.info('Replied to "mode" command ' + str(message.chat.id))
    return

@bot.callback_query_handler(func=lambda call: call.data.split('|')[0] == 'mode')
def changeItself(call):
    if int(call.data.split('|')[1]) in [0, 1, 2, 3]:
        modes[call.from_user.id] = int(call.data.split('|')[1]) 
        logging.info('Changed mode to ' + call.data.split('|')[1] + ' ' + str(call.from_user.id))
        bot.send_message(call.from_user.id, 'Changed mode to ' + call.data.split('|')[1])
    else:
        bot.send_message(call.from_user.id, 'Invalid mode, try again')
        logging.error('Not changed mode ' + call.data.split('|')[1] + ' ' + str(call.from_user.id))
    return
        
        
@bot.message_handler(func = lambda x : str(x.text)[0] != '/', content_types = ['text'])
def getText(message):
    if message.chat.id not in modes.keys():
        modes[message.chat.id] = 0
    os.makedirs(pic_path + str(message.chat.id), exist_ok=True)
    #text = ' '.join(list(map(lambda x : morph.parse(x)[0].normal_form, list(filter(lambda x: len(x) > 0, \
    #                                                                   str(message.text).split(' '))))))
    text = str(message.text).lower()
    logging.info('Received text ' + text + ' // ' + str(message.chat.id))
#    text = clean(text)
    opath = ''
    tpath = ''
    try:
        if modes[message.chat.id] in [0,1]:
            success = 0
            tries = 0
            while success!=1:
                tpath = text_rep_path + text.replace(' ','_') + '.npy'
                if not os.path.exists(tpath):
                    channel.basic_publish(exchange='', routing_key='toenc', body = str(text))
                    while not os.path.exists(tpath):
                        continue
                logging.info('Found file ' + tpath)
                opath = pic_path + str(message.chat.id) + '/' + text.replace(' ', '_') + '.png'
                captions = np.zeros((64,512))
                logging.info('Loaded text repr file ' + tpath)
                try:
                    captions[63, :] = np.load(tpath, allow_pickle = True)
                    success = 1
                except Exception as e:
                    tries += 1
                    if tries == 3:
                        raise(e)
                    logging.error('Error in loading npy ' + text + ' it is ' + str(tries) + 'try // ' + str(message.chat.id))
                    continue
                logging.info('Encoding made ' + str(message.chat.id))
                z_noise = np.random.uniform(-1, 1, [64, 100])
                val_feed = {
                    input_tensors['t_real_caption'].name: captions,
                    input_tensors['t_z'].name: z_noise,
                    input_tensors['t_training'].name: True
                }
                logging.info('Started generating ' + str(message.chat.id))
                scorelog = flowerlog.predict_proba(captions[63,:].reshape(1, -1))[0][1]
                logging.info('Text classifier ' + str(message.chat.id) + ' score ' + str(scorelog))
                if scorelog > 0.3:
                    val_gen = sessflowers.run(
                    [outputs['generator']],
                    feed_dict=val_feed)
                else:
                    val_gen = sessdefault.run(
                        [outputs['generator']],
                        feed_dict=val_feed)
                val_gen = np.squeeze(val_gen)
                fake_image_255 = val_gen[-1]
                imageio.imwrite(opath,fake_image_255)
                logging.info('Saved a pic to ' + opath)
        elif modes[message.chat.id] == 2:
            text = clean(text).replace(' ', '+')
            opath = ''
            resp = requests.get('https://go.mail.ru/search_images?fr=main&frm=main&q=' + text + '&fm=1').text
            resp = resp.split('"orig_url":"')
            resp_ind = np.random.randint(1,len(resp) - 1)
            resp = resp[resp_ind].split('"')[0]
            opath = pic_path + str(message.chat.id) + '/' + text.replace(' ','_') + '.png'
            wget.download(resp.strip(), out = opath, bar = None)
            logging.info('Saved a pic to ' + opath)
    except Exception as e:
        logging.error('Failed to get image // ' + str(message.chat.id) + ' // ' + str(e))
        bot.send_message(message.chat.id, 'Ooops, we cannot deliver you a picture, try again later please')
        return
    
    
    
    if modes[message.chat.id] in [0,2]:
        stylizationInit(message, opath)
        return
    try:
        bot.send_photo(message.chat.id, photo = open(opath, 'rb'))
    except Exception as e:
        logging.error('Failed to send image // ' + str(message.chat.id) + ' // ' + str(e))
        bot.send_message(message.chat.id, 'Ooops, we cannot deliver you a picture, try again later please')
        bot.register_next_step_handler(message, getText)
        return
    logging.info('Sent image to //' + str(message.chat.id))
    return

def stylizationInit(message, opath):
    logging.info('Starting stylization //' + str(message.chat.id))
    keyboard = types.InlineKeyboardMarkup(row_width = 1)
    keyboard.row(types.InlineKeyboardButton(text='Choose style', callback_data='selection|0|' + opath))
    keyboard.row(types.InlineKeyboardButton(text='Upload style image', callback_data='selection|1|' + opath))
    bot.send_message(message.chat.id, text = 'Select stylization type', reply_markup = keyboard)
    logging.info('Sent style type choosing to ' + str(message.chat.id))
    return

@bot.callback_query_handler(func = lambda call: call.data.split('|')[0] == 'selection')
def styleTypeSelection(call):
    opath = call.data.split('|')[2]
    if call.data.split('|')[1] == '0':
        keyboard = types.InlineKeyboardMarkup(row_width = 1)
        for i,key in enumerate(stylesdct.keys()):
            keyboard.row(types.InlineKeyboardButton(text=str(key), callback_data='style|' + str(i) + '|' + opath))
        bot.send_message(call.message.chat.id, text = 'Choose style', reply_markup=keyboard)
        logging.info('Sent style keyboard to ' + str(call.message.chat.id))
    else:
        bot.send_message(call.message.chat.id, 'Send me a style picture')
        bot.register_next_step_handler(call.message, ownStyleInference)
        ownstyle_transfer_dict[call.message.chat.id] = opath
        logging.info('Send style image request to ' + str(call.message.chat.id))
    return

def ownStyleInference(message):
    opath = ownstyle_transfer_dict[message.chat.id]
    file_id = message.photo[-1].file_id
    with open(own_styles_path + str(file_id) +  '.jpg', 'wb') as imout:
        imout.write(bot.download_file(bot.get_file(file_id).file_path))
    logging.info('Saved style img ' + str(file_id))
    style_img = image_loader(own_styles_path + str(file_id) +  '.jpg')
    content_img = image_loader(opath)
    output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                            content_img, style_img, content_img.clone())
    output = output.squeeze(0)      
    output = unloader(output)
    output.save(opath)
    try:
        bot.send_photo(message.chat.id, photo = open(opath, 'rb'))
    except Exception as e:
        logging.error('Failed to send image // ' + str(message.chat.id) + ' // ' + str(e))
        bot.send_message(message.chat.id, 'Ooops, we cannot deliver you a picture, try again later please')
        bot.register_next_step_handler(message, getText)
        return
    logging.info('Sent image to //' + str(message.chat.id))
    return

@bot.message_handler(func = lambda x : True, content_types = ['photo'])
def ownPicStylization(message):
    if modes[message.chat.id] != 3:
        bot.send_message(message.chat.id, 'Send text description or choose another mode')
        return
    opath = own_pics_path + str(len(os.listdir(own_pics_path))) + '.png'
    file_id = message.photo[-1].file_id
    with open(opath, 'wb') as imout:
        imout.write(bot.download_file(bot.get_file(file_id).file_path))
    logging.info('Saved content img ' + opath)
    stylizationInit(message, opath)
    return

        
@bot.callback_query_handler(func = lambda call: call.data.split('|')[0] == 'style')
def styleCallback(call):
    _, ind, opath = call.data.split('|')
    sind = np.random.randint(0,len(stylesdct))
    while sind == int(ind):
        sind = np.random.randint(0,len(stylesdct))
    try:
        im = Image.open(opath).convert('RGB')
        infer = inference.eval_image(im, int(ind), sind, 0.8)
        infer.save(opath)
    except Exception as e:
        logging.error('Failed to stylize image // ' + str(call.message.chat.id) + ' // ' + str(e))
        bot.send_message(call.message.chat.id, 'Ooops, we cannot deliver you a picture, try again later please')
        bot.register_next_step_handler(call.message, getText)
        return
        
        logging.info('Finished stylization //' + str(message.chat.id))
        
    try:
        bot.send_photo(call.message.chat.id, photo = open(opath, 'rb'))
    except Exception as e:
        logging.error('Failed to send image // ' + str(call.message.chat.id) + ' // ' + str(e))
        bot.send_message(call.message.chat.id, 'Ooops, we cannot deliver you a picture, try again later please')
        bot.register_next_step_handler(call.message, getText)
        return
    logging.info('Sent image to //' + str(call.message.chat.id))
    return







###################################ADDITIONAL############################################
    
@bot.message_handler(commands = ['notify'])
def broadcast(message):
    
    logging.info('Broadcast received')
    
    if message.from_user.username != 'eles13':
        return
    try:
        bot.send_message(message.chat.id, 'Ready for input')
        bot.register_next_step_handler(message, broadcasting_itself)
    except Exception as e:
        logging.error('Broadcasting start failed // ' + str(e))

def broadcasting_itself(message):
    logging.info('Broadcast starting')
    for idd in os.listdir(pic_path):
        try:
            bot.send_message(int(idd), message.text)
        except Exception as e:
            logging.error('Failed to deliver to ' + idd)
            bot.send_message(message.chat.id, 'Failed to deliver to ' + idd)
            continue
    logging.info('Broadcast finished')
    
@bot.message_handler(commands = ['change'])
def change_model_init(message):
    logging.info('Change model request received')
    bot.send_message(message.chat.id, 'Ready for input, send a shared Google Drive link')
    logging.info('Replied to "change model" request')
    try:
        os.mkdir(all_models_path + 'model_' + str(len(os.listdir(all_models_path))))
        logging.info('Created directory for a new model')
    except:
        logging.error('Failed to create a directory for a new model')
        bot.send_message(message.chat.id, 'Try again, failed to create a directory for a new model')
        return
    bot.register_next_step_handler(message, download_and_change)
    return

def download_and_change(message):
    global gan
    link = str(message.text).split('/')[5]
    logging.info('Received model link ' + link)
    current_model_path = all_models_path + 'model_' + str(len(os.listdir(all_models_path)) - 1)
    logging.info('Starting downloading')
    bot.send_message(message.chat.id, 'Starting downloading')
    try:
        gdd.download_file_from_google_drive(file_id=link,
                                        dest_path=current_model_path + '/archive.zip',
                                        unzip=True)
    except Exception as e:
        bot.send_message(message.chat.id, 'Failed to download, send link again, exception ' + str(e))
        bot.register_next_step_handler(download_and_change)
        logging.error('Failed to download, exception ' + str(e))
        return
    logging.info('Finished downloading')
    gan = DCGan()
    gan.load_model(current_model_path)
    logging.info('Model changed')
    bot.send_message(message.chat.id, 'Successfully changed model to iteration ' + str(len(os.listdir(all_models_path)) - 1))
    return
    

bot.polling(none_stop=True, interval = 0)
