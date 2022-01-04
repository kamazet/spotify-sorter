Sorter gets authorized access to a user's Spotify account. Its scope grants the software to read and modify a user's Spotify library. It will display all of the playlists belonging to the current user and they wil be prompted to give an input of the name of the playlist they would like to sort. The software then will use a function based on machine learning by "cristobalvch" on Github (https://github.com/cristobalvch/Spotify-Machine-Learning) to find the mood of the song. The song will then be added into a new playlist titled by the mood.

There are a few limitations to this software. It relies on getting track IDs for each song in a playlist, but the Spotify API is limited to only return 100 items. Playlists with more than 100 songs will not be sorted completely. Another limitation to Sorter is that because its authentication process relies creating a My App on Spotify Developer, users can only use Sorter when their email is registered on My App on Spotify Developer.

Before beginning, set up environment variables.
export SPOTIPY_CLIENT_ID='d03ce4c46d094731a656380af0830b6c'
export SPOTIPY_CLIENT_SECRET='e835dd5fa3b646daaf84109ebbb1ce00'
export SPOTIPY_REDIRECT_URI='http://google.com/'

Video: https://youtu.be/Zhfg_hpQeEI