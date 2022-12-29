import streamlit as st

# Load the model
model = tf.keras.models.load_model('my_model.h5')

# Create a function to use the model for prediction
def predict(inputs):
  prediction = model.predict(inputs)
  return prediction

# Load the dataset
dataset = pd.read_excel("data_label.xlsx")
dataset = dataset[['Content','Label']]
dataset = dataset.rename(columns={'Content':'text','Label':'label'})

# Preprocess the dataset
vec = CountVectorizer().fit(dataset['text'])
vec_transform = vec.transform(dataset['text'])
x = vec_transform.toarray()
y = dataset['label']

# Split the dataset into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=21)

# Train the model
metodeNB = MultinomialNB().fit(x_train, y_train)

# Add a UI element to accept input
inputs = st.text_input("Enter input data")

# Use the model to make a prediction
if st.button("Predict"):
  output = predict(inputs)
  st.success(f"Prediction: {output}")

# Load the data for prediction
data_pred = pd.read_excel("data uji.xlsx")
data_pred = data_pred[['Content']]
data_pred = data_pred.rename(columns={'Content':'text'})

# Preprocess the data for prediction
data_pred['vector'] = data_pred['text'].astype(str)
vec = CountVectorizer().fit(data_pred['vector'])
vec_transform = vec.transform(data_pred['vector'])
x_test = vec_transform.toarray()

# Make predictions on the data
data_pred['sentimen'] = metodeNB.predict(x_test)

# Display the prediction results
st.write(data_pred['sentimen'].value_counts())

# Plot the prediction results
count = data_pred['sentimen'].value_counts()
plt.figure(figsize=(10, 6))
plt.bar(['Positive', 'Netral', 'Negative'], count, color=['royalblue','green', 'orange'])
plt.xlabel('Jenis Sentimen', size=14)
plt.ylabel('Frekuensi', size=14)
st.pyplot()

# Make predictions on the data
data_pred['sentimen'] = metodeNB.predict(x_test)

# Display the prediction results
st.write(data_pred['sentimen'].value_counts())

# Plot the prediction results
count = data_pred['sentimen'].value_counts()
plt.figure(figsize=(10, 6))
plt.bar(['Positive', 'Netral', 'Negative'], count, color=['royalblue','green', 'orange'])
plt.xlabel('Jenis Sentimen', size=14)
plt.ylabel('Frekuensi', size=14)
st.pyplot()

