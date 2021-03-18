style_transfer_dir = './Real-time-multi-style-transfer/'
superresolution = True
import telebot
import re
from telebot import TeleBot
from telebot import types
from time import sleep
import os
import codecs
#os.system('nohup python ./text_encoder.py &')
#os.system('nohup python ./sr_processor.py &')
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
from fake_useragent import UserAgent
from sklearn.linear_model import LogisticRegression
tf.compat.v1.disable_eager_execution()
from own_style import *
import codecs
from bs4 import BeautifulSoup
import os.path
import json
from io import BytesIO

connection = pika.BlockingConnection(pika.ConnectionParameters(heartbeat=0 ,socket_timeout=100000))
channel = connection.channel()
channel.queue_declare(queue='toenc')

if superresolution:
    connectionsr = pika.BlockingConnection(pika.ConnectionParameters(heartbeat=0 ,socket_timeout=100000))
    channelsr = connectionsr.channel()
    channelsr.queue_declare(queue='tosr')
#################################################################
#############################SETUP###############################
#################################################################
pic_path = './saved_pics/'
text_rep_path = './texts/'
own_styles_path = './own_styles/'
own_pics_path = './own_pics/'
review_dir = './reviews/'

modes = {}
images={}
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
os.makedirs(review_dir, exist_ok=True)

logging.basicConfig(filename="botlog.log", level=logging.INFO, filemode='w')
token = ''
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
sessbirds = tf.compat.v1.InteractiveSession()
sessanime = tf.compat.v1.InteractiveSession()
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
    
if tf.train.latest_checkpoint('./checkpoints/birds/') is not None:
    saver.restore(sessbirds, tf.train.latest_checkpoint('./checkpoints/birds/'))
    logging.info('Successfully loaded model from ./checkpoints/birds/')
else:
    logging.error('Could not load checkpoints. Please provide a valid path to'
          ' your checkpoints directory')
    exit()
    
if tf.train.latest_checkpoint('./checkpoints/anime/') is not None:
    saver.restore(sessanime, tf.train.latest_checkpoint('./checkpoints/anime/'))
    logging.info('Successfully loaded model from ./checkpoints/anime/')
else:
    logging.error('Could not load checkpoints. Please provide a valid path to'
          ' your checkpoints directory')
    exit()
# #################################################################






####################search image downloaders and helper code######################
def get_file(url):
    #print(url)
    with requests.get(url, 
                           # proxies=dict(http='socks5://127.0.0.1:9150',
                           #              https='socks5://127.0.0.1:9150'), 
                      headers={'User-Agent': UserAgent().chrome}) as r:
            r.raise_for_status()
            return r.content
    

    
def get_mailru_image(text, out):
    try:
        #text = clean(text).replace(' ', '+')
        
        resp = requests.get('https://go.mail.ru/search_images?fr=main&frm=main&q=' + text + '&fm=1',
                            headers={'User-Agent': UserAgent().chrome}, stream = True).text
        resp = resp.split('"orig_url":"')
        resp_ind = np.random.randint(1,len(resp) - 1)
        resp = resp[resp_ind].split('"')[0]
        #wget.download(resp.strip(), out = out, bar = None)
        img_data = get_file(resp.strip())
        stream = BytesIO(img_data)
        #need libwebp library and reinstall of pillow
        im = Image.open(stream)       
        im.save(out)
        return True
        
    except Exception as e:
        print("[2]", text, out, repr(e))
        return False    
    
def get_mailru_image_old(text, out):
    try:
        text = clean(text).replace(' ', '+')
        out = ''
        resp = requests.get('https://go.mail.ru/search_images?fr=main&frm=main&q=' + text + '&fm=1',
                            headers={'User-Agent': UserAgent().chrome}, stream = True).text
        resp = resp.split('"orig_url":"')
        resp_ind = np.random.randint(1,len(resp) - 1)
        resp = resp[resp_ind].split('"')[0]
        wget.download(resp.strip(), out = out, bar = None)
        
    except Exception as e:
        print("[2]", text, out, repr(e))
        return False
 
def get_shutterstock_tattoo_image(text, out):
    return get_shutterstock_image(text + "+tattoo", out)
    
def get_shutterstock_image(text, out):
    query = text# + "+" + "tattoo"
    for i in range(1, 2):
        try:
            r = requests.get('https://www.shutterstock.com/search/' +query + '?page=' + str(i), 
                            #proxies=dict(http='socks5://98.143.144.44:61110',
                            #             https='socks5://98.143.144.44:61110'), 
                             headers={'User-Agent': UserAgent().chrome})
        
            content = str(r.content)
            bm = "</script><script data-react-helmet=\"true\" type=\"application/ld+json\">"
            begin = content.find(bm)
            a = (content[begin + len(bm):])
            b = a.find("}]")
            data = (a[:b+2])
            
            j = json.loads(data)
            rand_ind = np.random.randint(0,min(20, len(j)))
            #print(type(j))
            for a in [j[rand_ind]] + j:
                try:
                    url = a['url']
                    title = a['name']
                    alp = a['acquireLicensePage']
                    img_data = get_file(url)
                    stream = BytesIO(img_data)
                    #need libwebp library and reinstall of pillow
                    im = Image.open(stream)
                    w, h = im.size
                    im = im.crop((0,0,w,h*0.9))
                    im.save(out)
                    print(url)
                    return True  #download ok
  
                except Exception as e:
                    print("[2]", text, out, repr(e))
                    return False
            #break
           
        except Exception as e:
           print("[2]", text, out, repr(e))
           return False

    return False




def get_duck_image(keywords, out):
    try:
        url = 'https://duckduckgo.com/';
        params = {
            'q': keywords
        };



        #   First make a request to above URL, and parse out the 'vqd'
        #   This is a special token, which should be used in the subsequent request
        res = requests.post(url, data=params)
        searchObj = re.search(r'vqd=([\d-]+)\&', res.text, re.M|re.I);

        if not searchObj:

            return False;

        #logger.debug("Obtained Token");

        headers = {
            'authority': 'duckduckgo.com',
            'accept': 'application/json, text/javascript, */*; q=0.01',
            'sec-fetch-dest': 'empty',
            'x-requested-with': 'XMLHttpRequest',
            'user-agent': UserAgent().chrome,
            'sec-fetch-site': 'same-origin',
            'sec-fetch-mode': 'cors',
            'referer': 'https://duckduckgo.com/',
            'accept-language': 'en-US,en;q=0.9',
        }

        params = (
            ('l', 'us-en'),
            ('o', 'json'),
            ('q', keywords),
            ('vqd', searchObj.group(1)),
            ('f', ',,,'),
            ('p', '1'),
            ('v7exp', 'a'),
        )

        requestUrl = url + "i.js";

       


        for i in range(2):
            try:
                res = requests.get(requestUrl, headers=headers, params=params);
                data = json.loads(res.text);
                break;
            except ValueError as e:
                #logger.debug("Hitting Url Failure - Sleep and Retry: %s", requestUrl);
                #time.sleep(5);
                #continue;
                return False


        #printJson(data["results"]);
        
        if len(data["results"])>=1:
            rand_ind = np.random.randint(0,min(20, len(data["results"])))
            res = data["results"][rand_ind]
            url = res["thumbnail"]
            #rint(url)

            img_data = get_file(url)
            stream = BytesIO(img_data)
            #need libwebp library and reinstall of pillow
            im = Image.open(stream)
            w, h = im.size
            #im = im.crop((0,0,w,h*0.9))
            im.save(out)
            #print(url)
            return True

        return False

    except Exception as e:
        print(repr(e))
        return False

#################################
def doSuperresolution(opath, message):
    logging.info('Superresolution started // ' + str(message.chat.id))
    source_path = opath
    dest_path = opath.split('/')
    dest_path[-1] = 'sr_' + dest_path[-1]
    dest_path = '/'.join(dest_path)
    if os.path.isfile(dest_path):
        os.remove(dest_path)
    channelsr.basic_publish(exchange='', routing_key='tosr', body = str(opath))
   
    while not os.path.exists(dest_path):
        sleep(0.05)
        continue
    #modified sr_processor.py to write and then move to the path, no need of this delay
    sleep(0.06)
    logging.info('Superresolution ended // ' + str(message.chat.id) + "//" + str(opath) + "//" +  str(dest_path))
    im = Image.open(dest_path).convert('RGB')
    im.save(dest_path)
    return dest_path
    
def needSuperresolution(opath):
    im = Image.open(opath)
    w, h = im.size
    if (w<500) or (h<300):
        return True
    return False
    
    
#################################
@bot.message_handler(commands = ['start'])
def startBot(message):
    global running
    logging.info('Preparing to reply to "start" command ' + str(message.chat.id))
    bot.send_message(message.chat.id,'Привет, это бот АнтиЛебедев! Он умеет генерировать цветы и абстракции из текста, а также стилизовать их под 33 заданных стиля или под твой стиль. Этого бота можно использовать для получения красивых тату. Для начала набери команду /mode')
    logging.info('Replied to "start" command ' + str(message.chat.id))
    running = True
    return

@bot.message_handler(commands = ['mode'])
def changeMode(message):
    if message.chat.id not in modes.keys():
        modes[message.chat.id] = 0
    logging.info('Received "mode" command ' + str(message.chat.id))
    keyboard = types.InlineKeyboardMarkup(row_width = 1)
    keyboard.row(types.InlineKeyboardButton(text='генерация и стилизация', callback_data='mode|0'))
    keyboard.row(types.InlineKeyboardButton(text='генерация', callback_data='mode|1'))
    keyboard.row(types.InlineKeyboardButton(text='магия', callback_data='mode|2'))
    keyboard.row(types.InlineKeyboardButton(text='стилизация', callback_data='mode|3'))
    bot.send_message(message.chat.id, text = 'Выбери мод', reply_markup=keyboard)
    logging.info('Replied to "mode" command ' + str(message.chat.id))
    return

@bot.callback_query_handler(func=lambda call: call.data.split('|')[0] == 'mode')
def changeItself(call):
    if int(call.data.split('|')[1]) == 0:
        modes[call.from_user.id] = 0 
        logging.info('Changed mode to 0 ' +call.data.split('|')[1] + ' ' + str(call.from_user.id))
        bot.send_message(call.from_user.id, 'Сгенерируй картинку из текста, а затем стилизуй ее.')
    elif int(call.data.split('|')[1]) == 1:
        modes[call.from_user.id] = 1
        logging.info('Changed mode to 1 ' + call.data.split('|')[1] + ' ' + str(call.from_user.id))
        bot.send_message(call.from_user.id, 'Пиши текст, получай картинку! Наслаждайся цветами или абстрацией. Надо другими категориями мы усердно работаем')
    elif int(call.data.split('|')[1]) == 2:
        modes[call.from_user.id] = 2
        logging.info('Changed mode to 2 ' + call.data.split('|')[1] + ' ' + str(call.from_user.id))
        bot.send_message(call.from_user.id, 'Тут происходит магия, о которой мы не расскажем. Отправляй мне текст.')
    elif int(call.data.split('|')[1]) == 3:
        modes[call.from_user.id] = 3
        logging.info('Changed mode to 3 ' + call.data.split('|')[1] + ' ' + str(call.from_user.id))
        bot.send_message(call.from_user.id, 'Загрузи свое фото, а затем выбери стиль.')
    else:
        bot.send_message(call.from_user.id, 'Ты тупой???')
        logging.error('Not changed mode ' + call.data.split('|')[1] + ' ' + str(call.from_user.id))
    return
        
        
@bot.message_handler(func = lambda x : str(x.text)[0] != '/', content_types = ['text'])
def getText(message):
    if message.chat.id not in modes.keys():
        modes[message.chat.id] = 2
    os.makedirs(pic_path + str(message.chat.id), exist_ok=True)
    #text = ' '.join(list(map(lambda x : morph.parse(x)[0].normal_form, list(filter(lambda x: len(x) > 0, \
    #                                                                   str(message.text).split(' '))))))
    anime = False
    text = str(message.text).lower()
    if 'аниме' in text or 'anime' in text:
        anime = True
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
                if anime:
                    val_gen = sessanime.run(
                    [outputs['generator']],
                    feed_dict=val_feed)
                else:
                    scorelog = flowerlog.predict_proba(captions[63,:].reshape(1, -1))[0]
                    scorepred = flowerlog.predict(captions[63,:].reshape(1, -1))[0]
                    logging.info('Text classifier ' + str(message.chat.id) + ' score ' + str(scorelog))
                    if scorepred == 1 and scorelog <= 0.36 :
                        val_gen = sessflowers.run(
                        [outputs['generator']],
                        feed_dict=val_feed)
                    elif scorepred == 0 and scorelog[2] <= 0.3:
                        val_gen = sessbirds.run(
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
            opath = pic_path + str(message.chat.id) + '/' + text.replace(' ','_') + '.png'
            for image_downloader in [get_shutterstock_tattoo_image, get_shutterstock_image, get_mailru_image, get_duck_image]:
                if image_downloader(text, out=opath):
                    break
                else:
                    logging.error('Failed to get image using downloader ' +  str(image_downloader))
           
            
            logging.info('Saved a pic to ' + opath)
    except Exception as e:
        logging.error('Failed to get image // ' + str(message.chat.id) + ' // ' + str(e))
        bot.send_message(message.chat.id, 'Извини, что-то сломалось... Попробуй позже.')
        return
    
    
    
    if modes[message.chat.id] in [0,2]:
        stylizationInit(message, opath)
        return
    try:
        if superresolution:
            opath = doSuperresolution(opath, message)
        with open(opath, 'rb') as photo:
            bot.send_photo(message.chat.id, photo = photo)
    except Exception as e:
        logging.error('Failed to send image // ' + str(message.chat.id) + ' // ' + str(e))
        bot.send_message(message.chat.id, 'Извини, что-то сломалось... Попробуй позже.')
        bot.register_next_step_handler(message, getText)
        return
    logging.info('Sent image to //' + str(message.chat.id))
    return

def stylizationInit(message, opath):
    #opath = base64.b64encode(bytes(opath, "utf-8")).decode("utf-8")
    #opath = codecs.encode(bytes(opath, "utf-8"), "hex_codec").decode("utf-8")
    opath_hash = str(hash(opath))
    images[opath_hash] = opath
    #opath = opath_hash
    logging.info('Starting stylization //' + str(message.chat.id) + "//" + str(opath_hash))
    keyboard = types.InlineKeyboardMarkup(row_width = 1)
    keyboard.row(types.InlineKeyboardButton(text='Здесь 33 готовых стиля', callback_data='selection|0|' + opath_hash))
    keyboard.row(types.InlineKeyboardButton(text='Свой стиль', callback_data='selection|1|' + opath_hash))
    bot.send_message(message.chat.id, text = 'Выбор стиля', reply_markup = keyboard)
    logging.info('Sent style type choosing to ' + str(message.chat.id))
    return

@bot.callback_query_handler(func = lambda call: call.data.split('|')[0] == 'selection')
def styleTypeSelection(call):
    #opath = call.data.split('|')[2]
    opath_hash = call.data.split('|')[2]
    if opath_hash in images:
        opath = images[opath_hash]
    else:
        logging.error('Image not in images' + opath_hash)
        return
    #opath = base64.b64decode(opath).decode("utf-8")
    #opath = codecs.decode(opath, "hex_codec").decode("utf-8")
    if call.data.split('|')[1] == '0':
        keyboard = types.InlineKeyboardMarkup(row_width = 1)
        for i,key in enumerate(stylesdct.keys()):
            keyboard.row(types.InlineKeyboardButton(text=str(key), callback_data='style|' + str(i) + '|' + opath_hash))
        bot.send_message(call.message.chat.id, text = 'Выбор стиля', reply_markup=keyboard)
        logging.info('Sent style keyboard to ' + str(call.message.chat.id))
    else:
        bot.send_message(call.message.chat.id, 'Отправь картинку со своим стилем')
        bot.register_next_step_handler(call.message, ownStyleInference)
        ownstyle_transfer_dict[call.message.chat.id] = opath
        logging.info('Send style image request to ' + str(call.message.chat.id))
    return

def ownStyleInference(message):
    opath = ownstyle_transfer_dict[message.chat.id]
    try:
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
        if superresolution:
            opath = doSuperresolution(opath, message)
        with open(opath, 'rb') as photo:
            bot.send_photo(message.chat.id, photo = photo)
    except Exception as e:
        logging.error('Failed to send image // ' + str(message.chat.id) + ' // ' + str(e))
        bot.send_message(message.chat.id, 'Извини, что-то сломалось... Попробуй позже.')
        bot.register_next_step_handler(message, getText)
        return
    logging.info('Sent image to //' + str(message.chat.id))
    return

@bot.message_handler(func = lambda x : True, content_types = ['photo'])
def ownPicStylization(message):
    if message.chat.id not in modes:
        modes[message.chat.id ] = 2 #hotfix
    if modes[message.chat.id] != 3:
        bot.send_message(message.chat.id, 'Отправь текствое описание или выбери мод')
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
    _, ind, opath_hash = call.data.split('|')
    
    
    if opath_hash in images:
        opath = images[opath_hash]
    else:
        logging.error('Image not in images' + opath_hash)
        return
    
    sind = np.random.randint(0,len(stylesdct))
    while sind == int(ind):
        sind = np.random.randint(0,len(stylesdct))
    try:
        im = Image.open(opath).convert('RGB')
        opath = opath + "_st_" + str(ind) + ".png"
        infer = inference.eval_image(im, int(ind), sind, 0.8)
        infer.save(opath)
    except Exception as e:
        logging.error('Failed to stylize image // ' + str(call.message.chat.id) + ' // ' + str(e))
        bot.send_message(call.message.chat.id, 'Извини, что-то сломалось... Попробуй позже.')
        bot.register_next_step_handler(call.message, getText)
        return
        
    logging.info('Finished stylization //' + str(call.message.chat.id))
        
    try:
       
        if superresolution and needSuperresolution(opath):
            opath = doSuperresolution(opath, call.message)
        with open(opath, 'rb') as photo:
            bot.send_photo(call.message.chat.id, photo = photo)
 
            
    except Exception as e:
        logging.error('Failed to send image // ' + str(call.message.chat.id) + ' // ' + str(e))
        bot.send_message(call.message.chat.id, 'Извини, что-то сломалось... Попробуй позже.')
        bot.register_next_step_handler(call.message, getText)
        return
    logging.info('Sent image to //' + str(call.message.chat.id))
    return







###################################ADDITIONAL############################################


@bot.message_handler(commands = ['review'])
def initReview(message):
    logging.info('Received review request // ' + str(message.chat.id))
    try:
        os.makedirs(review_dir + str(message.from_user.username), exist_ok = True)
        bot.send_message(message.chat.id, 'Расскажи, как я тебе?')
        bot.register_next_step_handler(message, reviewHandler)
        logging.info('Replied to review request // ' + str(message.chat.id))
    except Exception as e:
        logging.error('Failed to reply to review request // ' + str(message.chat.id) + ' // ' + str(e))
        bot.send_message(message.chat.id, 'Извини, что-то сломалось... Попробуй позже.')
        return
    
def reviewHandler(message):
    logging.info('Starting writing review // ' + str(message.chat.id))
    try:
        loc_dir = review_dir + str(message.from_user.username) + '/'
        with open(loc_dir + str(len(os.listdir(loc_dir))) + '.txt', 'w') as fout:
            fout.write(str(message.text))
        logging.info('Review written // ' + str(message.chat.id))
        bot.send_message(message.chat.id, 'Спасибо за отзыв! Можешь продолжать пользоваться ботом :)')
    except Exception as e:
        logging.error('Failed to write review // ' + str(message.chat.id) + ' // ' + str(e))
        bot.send_message(message.chat.id, 'Извини, что-то сломалось... Попробуй позже.')
        return    
    
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


