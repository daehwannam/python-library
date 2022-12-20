
# Original source: https://www.geeksforgeeks.org/bloom-filters-introduction-and-python-implementation/
#
# Python 3 program to build Bloom Filter
# Install mmh3 and bitarray 3rd party module first
# pip install mmh3
# pip install bitarray
import math
import mmh3
from bitarray import bitarray
 
 
class BloomFilter(object):
 
    '''
    Class for Bloom filter, using murmur3 hash function
    '''
 
    def __init__(self, num_expected_items, false_positive_prob):
        '''
        num_expected_items : int
            Number of items expected to be stored in bloom filter
        false_positive_prob : float
            False Positive probability in decimal
        '''
        # False posible probability in decimal
        self.false_positive_prob = false_positive_prob
 
        # Size of bit array to use
        self.size = self.get_size(num_expected_items, false_positive_prob)
 
        # number of hash functions to use
        self.hash_count = self.get_hash_count(self.size, num_expected_items)
 
        # Bit array of given size
        self.bit_array = bitarray(self.size)
 
        # initialize all bits as 0
        self.bit_array.setall(0)
 
    def add(self, item):
        '''
        Add an item in the filter
        '''
        digests = []
        for i in range(self.hash_count):
 
            # create digest for given item.
            # i work as seed to mmh3.hash() function
            # With different seed, digest created is different
            digest = mmh3.hash(item, i) % self.size
            digests.append(digest)
 
            # set the bit True in bit_array
            self.bit_array[digest] = True
 
    def check(self, item):
        '''
        Check for existence of an item in filter
        '''
        for i in range(self.hash_count):
            digest = mmh3.hash(item, i) % self.size
            if self.bit_array[digest] == False:
 
                # if any of bit is False then,its not present
                # in filter
                # else there is probability that it exist
                return False
        return True
 
    @classmethod
    def get_size(self, n, p):
        '''
        Return the size of bit array(m) to used using
        following formula
        m = -(n * lg(p)) / (lg(2)^2)
        n : int
            number of items expected to be stored in filter
        p : float
            False Positive probability in decimal
        '''
        m = -(n * math.log(p))/(math.log(2)**2)
        return int(m)
 
    @classmethod
    def get_hash_count(self, m, n):
        '''
        Return the hash function(k) to be used using
        following formula
        k = (m/n) * lg(2)
 
        m : int
            size of bit array
        n : int
            number of items expected to be stored in filter
        '''
        k = (m/n) * math.log(2)
        return int(k)


def test():
    from random import shuffle

    n = 20 #no of items to add
    p = 0.05 #false positive probability

    bloomf = BloomFilter(n,p)
    print("Size of bit array:{}".format(bloomf.size))
    print("False positive Probability:{}".format(bloomf.false_positive_prob))
    print("Number of hash functions:{}".format(bloomf.hash_count))

    # words to be added
    word_present = ['abound','abounds','abundance','abundant','accessable',
                    'bloom','blossom','bolster','bonny','bonus','bonuses',
                    'coherent','cohesive','colorful','comely','comfort',
                    'gems','generosity','generous','generously','genial']

    # word not added
    word_absent = ['bluff','cheater','hate','war','humanity',
                   'racism','hurt','nuke','gloomy','facebook',
                   'geeksforgeeks','twitter']

    for item in word_present:
        bloomf.add(item)

    shuffle(word_present)
    shuffle(word_absent)

    test_words = word_present[:10] + word_absent
    shuffle(test_words)
    for word in test_words:
        if bloomf.check(word):
            if word in word_absent:
                print("'{}' is a false positive!".format(word))
            else:
                print("'{}' is probably present!".format(word))
        else:
            print("'{}' is definitely not present!".format(word))
