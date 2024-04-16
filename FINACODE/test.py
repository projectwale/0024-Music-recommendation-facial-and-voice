# https://freefrontend.com/css-music-players/

import spotipy
from spotipy.oauth2 import SpotifyOAuth
from spotipy.oauth2 import SpotifyClientCredentials
from flask import Flask,render_template,request,session, url_for, redirect
#from flask_mysqldb import MySQL
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


app=Flask(__name__)
app.secret_key = 'random string'

#cid = 'eb9d53df11484a72b61c48259ffd7c8b'
#secret = '33f5f097a1404c8082bf1ce05b9c23bf'
cid="0f0b1633f7f74a0c937e46d42de6497c"
secret="4fd5e59894cf459686ff971fd0731b6a"

#spotify = spotipy.Spotify(auth_manager = SpotifyOAuth(client_id = cid,
                                                      #client_secret =secret,
                                                      #redirect_uri = 'http://localhost:5000/callback'))


auth_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret)
spotify = spotipy.Spotify(auth_manager=auth_manager)
def valid_token(resp):
    return resp is not None and not 'error' in resp


def search_all(search_type, name):
    print(name) 
    data = spotify.search(name,limit=50,type="track")   

    api_url = data['tracks']['items']
    items = data['tracks']['items']

    p=[]
    for i in range(len(items)):
        b=items[i]['id']
        p.append(b)
    import random
    items = random.sample(items, len(items))

    return name,items,api_url, search_type

@app.route("/", methods=['POST','GET'])
def search1():
    if request.method=="POST":
        search_type = request.form.get("search_type")
        name = request.form.get("name")

        print(search_type, name)

        name,items,api_url, search_type = search_all(search_type, name)

        return render_template('search.html',name=name,results=items,api_url=api_url, search_type=search_type)
    return render_template('search.html')

def make_search2(search_type, name):
    data = spotify.search(name,limit=20,type="track")   
    api_url = data['tracks']['items']
    items = data['tracks']['items']
    #print(items)
    p=[]
    for i in range(len(items)):
        b=items[i]['id']
        p.append(b)
    import random
    items = random.sample(items, len(items))
    dst=os.listdir("static/uploadeddetected")
    k=dst[0]

    return render_template('search2.html',
                           name=name,
                           results=items,
                           api_url=api_url,
                           k=k,
                           search_type=search_type)


if __name__=="__main__":
    # app.run("0.0.0.0")
    app.run(debug=True)