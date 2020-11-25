import sys
sr_dir = '../super_resolution/image-super-resolution/'
if sr_dir not in sys.path:
    sys.path.append(sr_dir)
from ISR.models import RDN
import pika
import numpy as np
from PIL import Image
rdn = RDN(weights='psnr-small')
def callback(ch, method, properties, body):
    opath = str(body.decode('utf8'))
    message = opath
    opath = opath.split('/')
    opath[-1] = 'sr_' + opath[-1]
    opath = '/'.join(opath)
    Image.fromarray(rdn.predict(np.array(Image.open(message)))).save(opath)
    

connection = pika.BlockingConnection(pika.ConnectionParameters(heartbeat=0 ,socket_timeout=100000))
channel = connection.channel()

channel.queue_declare(queue='tosr')

channel.basic_consume(queue= 'tosr', on_message_callback=callback, auto_ack=True)

channel.start_consuming()
