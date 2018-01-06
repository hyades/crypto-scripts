import requests
import urlparse
import numpy as np
from multiprocessing.pool import Pool
from pprint import pprint



class API(object):
    HOST = 'https://api.binance.com/api/v1/depth?symbol='
    ORDERBOOK_ENDPOINT = '/api/v1/depth'
    TICKER_ENDPOINT = '/api/v1/ticker/allPrices'

    def __init__(self):
        pass

    def make_get_request(self, url, payload):
        response = requests.get(url, params=payload)
        return response.json()

    def get_orderbook(self, symbol):
        url = urlparse.urljoin(self.HOST, self.ORDERBOOK_ENDPOINT)
        payload = {'symbol': symbol}
        return self.make_get_request(url, payload)

    def get_ticker(self):
        url = urlparse.urljoin(self.HOST, self.TICKER_ENDPOINT)
        payload = {}
        return self.make_get_request(url, payload)


class SymbolList(object):

    def __init__(self, raw_api_output):
        self.raw_api_output = raw_api_output
        self.symbols = list()
        self._digest()

    def _digest(self):
        for point in self.raw_api_output:
            if point['symbol'] == "BTCUSDT":
                self.btcprice = float(point['price'])
            if point['symbol'] == "ETHUSDT":
                self.ethprice = float(point['price'])
            if any(i.isdigit() for i in point['symbol']) or point['symbol'].endswith('BNB') or point['symbol'].endswith('BNB'):
                continue
            self.symbols.append((point['symbol'], float(point['price'])))
        print self.btcprice
        print self.ethprice

    def __iter__(self):
        for i in self.symbols:
            yield (i, self.btcprice, self.ethprice)


class Bid(object):

    def __init__(self, price, amount, symbol, btcprice, ethprice):
        self.price = price
        self.amount = amount
        self.symbol = symbol
        self.btcprice = btcprice
        self.ethprice = ethprice
        self.usd_price = self._calculate_usd_price()
        

    def _calculate_usd_price(self):
        if self.symbol.endswith('BTC'):
            return self.btcprice * self.price * self.amount
        if self.symbol.endswith('ETH'):
            return self.ethprice * self.price * self.amount
        return 0

    def __str__(self):
        return 'Bid(symbol: {0}, price: {1}, amount: {2}, usd: {3})'.format(self.symbol, self.price, self.amount, self.usd_price)

    def __repr__(self):
        return self.__str__()


class Ask(object):

    def __init__(self, price, amount):
        self.price = price
        self.amount = amount

    def __str__(self):
        return 'Ask(price: {0}, amount: {1})'.format(self.price, self.amount)

    def __repr__(self):
        return self.__str__()


class OrderBook(object):

    def __init__(self, raw_api_output, symbol, btcprice, ethprice):
        self.raw_api_output = raw_api_output
        self.symbol = symbol
        self.btcprice = btcprice
        self.ethprice = ethprice
        self.bids = []
        self.asks = []
        self._digest()

    def _digest(self):
        for raw_bid in self.raw_api_output['bids']:
            bid = Bid(
                price=float(raw_bid[0]),
                amount=float(raw_bid[1]),
                symbol=self.symbol,
                btcprice=self.btcprice,
                ethprice=self.ethprice,
                )
            self.bids.append(bid)

        for raw_ask in self.raw_api_output['asks']:
            ask = Ask(price=float(raw_bid[0]), amount=float(raw_bid[1]))
            self.asks.append(ask)


class WhaleClassfier(object):

    def __init__(self, bids, current_price):
        self.bids = bids
        self.current_price = current_price
        self.whales = []
        self._preprocess()
        self._process()

    def _is_abnormal_bid_size(self):
        thres = 100
        points = np.array([bid.amount for bid in self.bids])
        median = np.median(points, axis=0)
        diff = np.abs(points - median)
        med_abs_deviation = np.median(diff)

        modified_z_score = 0.6745 * diff / med_abs_deviation
        return modified_z_score > thres

    def _preprocess(self):
        self.bids = [x for x in self.bids if x.price > 0.99 * self.current_price]

    def _process(self):
        if len(self.bids) < 5:
            return

        funcs = [
            self._is_abnormal_bid_size
        ]

        whale = [True] * len(self.bids)

        for func in funcs:
            for index, result in enumerate(func()):
                whale[index] &= bool(result)

        for x, y in zip(whale, self.bids):
            if x is True:
                self.whales.append(y)


if __name__ == '__main__':
    api = API()

    def execute(elem):
        try:
            btcprice = elem[1]
            ethprice = elem[2]
            elem = elem[0]
            symbol = elem[0]
            price = elem[1]
            orderbook = OrderBook(api.get_orderbook(symbol), symbol, btcprice, ethprice)
            return {'whales': WhaleClassfier(orderbook.bids, price).whales, 'symbol': symbol, 'price': price}
        except Exception as e:
            print 'error', elem, e

    pool = Pool(64)
    result = pool.map(execute, SymbolList(api.get_ticker()))
    f_result = filter(lambda x: len(x['whales']) > 0, result)
    pprint(f_result)
