import os
import sys
import spotipy
import spotipy.util as util
from spotipy.oauth2 import SpotifyOAuth
from json.decoder import JSONDecodeError

#Script to obtain data 
import numpy as np 
import pandas as pd 

#Libraries to create the multiclass model
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils

#Import tensorflow and disable the v2 behavior and eager mode
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
tf.compat.v1.disable_v2_behavior()

#Library to validate the model
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from sklearn.pipeline import Pipeline

def base_model():
    #Create the model
    model = Sequential()
    #Add 1 layer with 8 nodes,input of 4 dim with relu function
    model.add(Dense(8,input_dim=10,activation='relu'))
    #Add 1 layer with output 3 and softmax function
    model.add(Dense(4,activation='softmax'))
    #Compile the model using sigmoid loss function and adam optim
    model.compile(loss='categorical_crossentropy',optimizer='adam',
                 metrics=['accuracy'])
    return model

def get_songs_features(ids):

    meta = sp.track(ids)
    features = sp.audio_features(ids)

    # meta
    name = meta['name']
    album = meta['album']['name']
    artist = meta['album']['artists'][0]['name']
    release_date = meta['album']['release_date']
    length = meta['duration_ms']
    popularity = meta['popularity']
    ids =  meta['id']

    # features
    acousticness = features[0]['acousticness']
    danceability = features[0]['danceability']
    energy = features[0]['energy']
    instrumentalness = features[0]['instrumentalness']
    liveness = features[0]['liveness']
    valence = features[0]['valence']
    loudness = features[0]['loudness']
    speechiness = features[0]['speechiness']
    tempo = features[0]['tempo']
    key = features[0]['key']
    time_signature = features[0]['time_signature']

    track = [name, album, artist, ids, release_date, popularity, length, danceability, acousticness,
            energy, instrumentalness, liveness, valence, loudness, speechiness, tempo, key, time_signature]
    columns = ['name','album','artist','id','release_date','popularity','length','danceability','acousticness','energy','instrumentalness',
                'liveness','valence','loudness','speechiness','tempo','key','time_signature']
    return track,columns

def predict_mood(id_song):
    #Join the model and the scaler in a Pipeline
    pip = Pipeline([('minmaxscaler',MinMaxScaler()),('keras',KerasClassifier(build_fn=base_model,epochs=300,batch_size=200,verbose=0))])
    #Fit the Pipeline
    pip.fit(X2,encoded_y)

    #Obtain the features of the song
    preds = get_songs_features(id_song)
    #Pre-process the features to input the Model
    preds_features = np.array(preds[0][6:-2]).reshape(-1,1).T

    #Predict the features of the song
    results = pip.predict(preds_features)

    mood = np.array(target['mood'][target['encode']==int(results)])

    return mood[0].upper()

# Find and list all playlists
def current_playlists():
    all_playlists = {}
    playlists = sp.current_user_playlists()
    playlist_length = playlists['total']
    for i in range(playlist_length):
        playlist_name = playlists['items'][i]['name'].upper().strip()
        playlist_id = playlists['items'][i]['id']
        all_playlists[playlist_name] = playlist_id
    
    return all_playlists

# https://github.com/cristobalvch/Spotify-Machine-Learning/blob/master/Keras-Classification.ipynb
df = pd.read_csv("data_moods.csv")
col_features = df.columns[6:-3]
X= MinMaxScaler().fit_transform(df[col_features])
X2 = np.array(df[col_features])
Y = df['mood']

#Encode the categories
encoder = LabelEncoder()
encoder.fit(Y)
encoded_y = encoder.transform(Y)

#Convert to  dummy (Not necessary in my case)
dummy_y = np_utils.to_categorical(encoded_y)

X_train,X_test,Y_train,Y_test = train_test_split(X,encoded_y,test_size=0.2,random_state=15)

target = pd.DataFrame({'mood':df['mood'].tolist(),'encode':encoded_y}).drop_duplicates().sort_values(['encode'],ascending=True)
target

# Get the username from terminal
username = sys.argv[1]
if len(sys.argv) != 2:
    print("Usage: python sorter.py USERNAME")
scope = 'user-library-read user-read-private user-read-playback-state user-modify-playback-state playlist-modify-private playlist-read-collaborative playlist-read-private playlist-modify-public'
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(scope=scope))

# Gets a csv of all user's playlists
playlists_csv = ", ".join(current_playlists().keys())
print(f"Here are all your playlists: {playlists_csv}")
input_playlist = input("Which playlist would you like to sort? ")
# Check if playlist is valid
# upper and strip work but @ line 149, selected_playlist = input_playlist isn't formatted right
if input_playlist.upper().strip() in playlists_csv.upper().strip():
    selected_playlist = input_playlist.upper().strip()
else:
    print("Sorry, that playlist doesn't exist :(")
    exit()

# Gets all tracks in playlist
selected_playlist_id = current_playlists()[selected_playlist]
tracks_ids = []
tracks = sp.playlist_tracks(playlist_id = selected_playlist_id, limit = 100)
for track in tracks['items']:
    tracks_ids.append(track['track']['id'])

# Creating playlists and lists
sp.user_playlist_create(user = username, name = "Calm")
sp.user_playlist_create(user = username, name = "Happy")
sp.user_playlist_create(user = username, name = "Energetic") 
sp.user_playlist_create(user = username, name = "Sad")
calm = []
happy = []
energetic = []
sad = []

# Gets features from tracks and organizes them in lists based on mood
for track_id in tracks_ids:
    mood = predict_mood(id_song = track_id)
    if mood == 'CALM':
        calm.append(track_id)
    if mood == 'HAPPY':
        happy.append(track_id)
    if mood == 'ENERGETIC':
        energetic.append(track_id)
    if mood == 'SAD':
        sad.append(track_id)

# Creates playlists based on mood lists
if len(calm) > 0:
    sp.playlist_add_items(playlist_id = current_playlists()["CALM"], items = calm)
if len(happy) > 0:
    sp.playlist_add_items(playlist_id = current_playlists()["HAPPY"], items = happy)
if len(energetic) > 0:
    sp.playlist_add_items(playlist_id = current_playlists()["ENERGETIC"], items = energetic)
if len(sad) > 0:
    sp.playlist_add_items(playlist_id = current_playlists()["SAD"], items = sad)