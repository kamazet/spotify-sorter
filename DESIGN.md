Starting on line 130, I ensure that the command line argument is correct, essentially taking in the user's input for their username. I also set up the scope to be used for authentication. That function goes under "sp" and would be called everytime I use Spotipy, a Spotify python library.

Once I get access to a user's library, I'm able to list out all of their playlist names as comma separated values. I print them and ask for a user's input to select a playlist. I make sure to check that the playlist that the user's inputted exists.

Using the nae of the playlist, I find its playlist ID and get all of the track IDs in that playlists and put them into an array so that I can index through it if I ever need to.

I make four arrays to store all the track IDs for once they are sorted.

I go through each track ID from the track ID array and predict the mood for each track. THe predict mood function is by "cristobalvch" on Github (https://github.com/cristobalvch/Spotify-Machine-Learning). I then check the result of the function to see what mood it matches up to. By that, I add that track ID to its corresponding mood array.

I check if each of the array is empty. If they aren't, I create that corresponding playlist and add the songs from each mood array into its respective playlist.