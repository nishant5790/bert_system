import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Embedding, LayerNormalization, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
# Define the BERT-like model architecture
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [Dense(ff_dim, activation="relu"), Dense(embed_dim),]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)
        
    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
class BertModel(tf.keras.Model):
    def __init__(self, vocab_size, max_len, embed_dim, num_heads, ff_dim, num_layers, num_classes):
        super(BertModel, self).__init__()
        self.token_embeddings = Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_embeddings = Embedding(input_dim=max_len, output_dim=embed_dim)
        self.transformer_blocks = [TransformerBlock(embed_dim, num_heads, ff_dim) for _ in range(num_layers)]
        self.dropout = Dropout(0.1)
        self.classifier = Dense(num_classes, activation="softmax")
        
    def call(self, inputs, training):
        max_len = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=max_len, delta=1)
        token_embeddings = self.token_embeddings(inputs)
        pos_embeddings = self.pos_embeddings(positions)
        x = token_embeddings + pos_embeddings
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, training)
        x = self.dropout(x, training)
        return self.classifier(x)
def create_input_data(texts, vocab, max_len):
    input_ids = []
    for text in texts:
        input_ids.append([vocab.get(word, vocab["<UNK>"]) for word in text.split()])
    input_ids = pad_sequences(input_ids, maxlen=max_len, padding='post')
    return input_ids
# Parameters
VOCAB_SIZE = 30522    # Typical BERT vocab size
MAX_LEN = 128         # Max sequence length
EMBED_DIM = 128       # Embedding size for each token
NUM_HEADS = 8         # Number of attention heads
FF_DIM = 128          # Hidden layer size in feed-forward network inside transformer
NUM_LAYERS = 2        # Number of transformer blocks
NUM_CLASSES = 4       # Number of output labels, including NER tags
# Create a simple vocabulary
vocab = {"<PAD>": 0, "<UNK>": 1, "my": 2, "name": 3, "is": 4, "nishant": 5, "and": 6, "i": 7, "am": 8, 
         "a": 9, "data": 10, "scientist": 11, "use": 12, "python": 13}
reverse_vocab = {v: k for k, v in vocab.items()}
# Create sample data
texts = ["my name is nishant and i am a data scientist", "i use python"]
labels = [[1, 2, 1, 3, 1, 1, 1, 1, 1, 4, 1], [1, 1, 4]]
# Create input and output data
X_train = create_input_data(texts, vocab, MAX_LEN)
y_train = pad_sequences(labels, maxlen=MAX_LEN, padding='post')
# Create and compile BERT-like model
bert = BertModel(VOCAB_SIZE, MAX_LEN, EMBED_DIM, NUM_HEADS, FF_DIM, NUM_LAYERS, NUM_CLASSES)
bert.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# Train the model
bert.fit(X_train, y_train, epochs=10, batch_size=2)
# Save the model
bert.save('bert_ner_model')
# Load the model
loaded_model = tf.keras.models.load_model('bert_ner_model', custom_objects={'BertModel': BertModel, 'TransformerBlock': TransformerBlock})
# Sample inference
def predict(text, model, vocab, max_len):
    input_data = create_input_data([text], vocab, max_len)
    predictions = model.predict(input_data)
    predicted_labels = np.argmax(predictions[0], axis=-1)
    return predicted_labels
sample_text = "my name is nishant and i am a data scientist"
predicted_labels = predict(sample_text, loaded_model, vocab, MAX_LEN)
print("Predicted labels:", predicted_labels)