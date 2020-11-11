import tensorflow_hub as hub
import pika
import numpy as np
model = hub.load('./tfhub/')
text_repr_path = './texts/'

def callback(ch, method, properties, body):
    message = str(body.decode('utf8'))
    np.save(text_repr_path + message.replace(' ', '_') + '.npy', model([message +' i']).numpy(), allow_pickle=False)
    

connection = pika.BlockingConnection(pika.ConnectionParameters())
channel = connection.channel()

channel.queue_declare(queue='toenc')

channel.basic_consume(queue= 'toenc', on_message_callback=callback, auto_ack=True)

channel.start_consuming()
