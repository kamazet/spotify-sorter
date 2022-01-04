import os
import sys
import spotipy
import spotipy.util as util
from json.decoder import JSONDecodeError

#Import tensorflow and disable the v2 behavior and eager mode
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
tf.compat.v1.disable_v2_behavior()

# Get the username from terminal
username = sys.argv[1]
scope = 'user-library-read user-read-private user-read-playback-state user-modify-playback-state playlist-modify-private playlist-read-collaborative playlist-read-private playlist-modify-public'

# Erase cache and prompt for user permission
try:
    token = util.prompt_for_user_token(username, scope) # add scope
except (AttributeError, JSONDecodeError):
    os.remove(f".cache-{username}")
    token = util.prompt_for_user_token(username, scope) # add scope

# Create our spotify object with permissions
sp = spotipy.Spotify(auth=token)

def current_playlists():
    all_playlists = {}
    playlists = sp.current_user_playlists()
    playlist_length = playlists['total']
    for i in range(playlist_length):
        playlist_name = playlists['items'][i]['name']
        playlist_id = playlists['items'][i]['id']
        all_playlists[playlist_name] = playlist_id
    
    return all_playlists

sp.current_user_unfollow_playlist(playlist_id = current_playlists()["Sad"])
sp.current_user_unfollow_playlist(playlist_id = current_playlists()["Happy"])
sp.current_user_unfollow_playlist(playlist_id = current_playlists()["Energetic"])
sp.current_user_unfollow_playlist(playlist_id = current_playlists()["Calm"])