import requests
import pandas as pd
from io import StringIO
import time
from datetime import datetime, timedelta
import sqlite3

conn = sqlite3.connect('data/yahoo.db')
cur = conn.cursor()

# first scrap
# url = 'https://query1.finance.yahoo.com/v7/finance/download/BTC-USD?period1=1410912000&period2=1591056000&interval=1d&events=history'
#
# r = requests.get(url)
# data = r.text
#
# yahoo = pd.read_csv(StringIO(data))
#
# cur.execute('create table btc ' + str(tuple(yahoo.columns.values)) + ';')
# cur.executemany('insert into btc ' + str(tuple(yahoo.columns.values)) + ' values (' + ('?,'* len(yahoo.columns))[:-1] + ')', yahoo.values)
#
# conn.commit()
#
# conn.close()

rows = cur.execute('select * from btc').fetchall()

col, *_ = zip(*cur.description)

yahoo = pd.DataFrame(data=rows, columns=col)

last = yahoo.Date.values[-1]

laststamp = int(datetime.timestamp(datetime.strptime(last, '%Y-%m-%d')))
nowstamp = int(datetime.timestamp(datetime.now()))

url = f'https://query1.finance.yahoo.com/v7/finance/download/BTC-USD?period1={laststamp}&period2={nowstamp}&interval=1d&events=history'

r = requests.get(url)
data = r.text

new_yahoo = pd.read_csv(StringIO(data))

new_yahoo = new_yahoo[new_yahoo.Date > last]

cur.executemany('insert into btc ' + str(tuple(new_yahoo.columns.values)) + ' values (' + ('?,'* len(new_yahoo.columns))[:-1] + ')', new_yahoo.values)

conn.commit()

conn.close()





















'''___'''
