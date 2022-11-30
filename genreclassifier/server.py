# using flask_restful
from flask import Flask, jsonify
from flask_restful import Resource, Api, request
import tensorflow as tf
import tensorflow_text as text
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

# creating the flask app
app = Flask(__name__)
# creating an API object
api = Api(app)


# another resource to calculate the square of a number
class SmallBert(Resource):

    def __init__(self):
        self.reloaded_model = tf.saved_model.load('../models/small_bert/bert_en_uncased_L-2_H-128_A-2')

    def get(self):
        processed_description = request.args['processed_description']
        probabilities = self.reloaded_model(tf.constant([processed_description])).numpy().tolist()[0]
        probabilities_dict = dict(zip(range(len(probabilities)), probabilities))
        return jsonify({'probabilities': probabilities_dict})


# adding the defined resources along with their corresponding urls
api.add_resource(SmallBert, '/small_bert/get_probabilities')

# driver function
if __name__ == '__main__':
    app.run(debug=True)
