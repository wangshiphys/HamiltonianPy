"""Provide functions to encrypt and decrypt a string
"""

__all__ = ['encrypt', 'decrypt']

import base64
import random

# The ceiling of a byte
# Byte should be in the range [0, 255], no larger than 256
BYTE_CEIL = 256

# The encoding used to convert between strings and bytes
CODING = "utf-8"

# URL- and filesystem-safe base64 alphabet which substitutes '-' instead of '+'
# and '_' instead of '/' in the standard base64 alphabet.
BASE64_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"


def encrypt(msg, key=None):# {{{
    """
    Return an encrypted version of the 'msg' string according to the given 'key'

    Parameters
    ----------
    msg : str
        A string that is to be encrypted
    key : str, optional
        A string that is used to encrypt the 'msg'
        If not given or None, a random string with the same length as 'msg' is
        generated using the BASE64_ALPHABET.
        default : None

    Returns
    -------
    enc_msg : str
        The encrypted version of 'msg'
        This is the first entry of the returning tuple.
    key : str
        The string used to encrypt 'msg'
        This is the second entry of the returning tuple.

    Examples
    --------
    >>> from cryptography import encrypt
    >>> encrypt(msg="abcde", key="edcba")
    ('xsbGxsY=', 'edcba')
    """

    msg_bytes = msg.encode(encoding=CODING)
    msg_len = len(msg_bytes)

    if key is None:
        key = "".join(random.sample(BASE64_ALPHABET, msg_len))
    key_bytes = key.encode(encoding=CODING)
    key_len = len(key_bytes)

    enc_bytes = bytes((msg_bytes[i] + key_bytes[i%key_len]) % BYTE_CEIL
            for i in range(msg_len))
    enc_msg = base64.urlsafe_b64encode(enc_bytes).decode(encoding=CODING)

    while True:
        ans = input(
            "Do you want to write the encrypted message to a file ([y]/n)? ")
        if ans.lower() in ('', 'y', "yes"):
            fp = open("encrypted_message", 'w')
            fp.write(enc_msg + '\n')
            fp.close()
            fp = open("private_key", 'w')
            fp.write(key + '\n')
            fp.close()
            break
        elif ans.lower() in ('n', "no"):
            break
        else:
            print("Please input y[es] or n[o]!")
    return enc_msg, key
# }}}


def decrypt(msg, key):# {{{
    """
    Return an decrypted version of the 'msg' string according to the given 'key'

    Parameters
    ----------
    msg : str
        The string to be decrypted
    key : str
        The string used to decrypted the given 'msg'

    Returns
    -------
    msg : str
        The decrypted version of 'msg'

    Examples
    --------
    >>> from cryptography import decrypt, encrypt
    >>> decrypt(*encrypt("abcde", "edcba"))
    'abcde'
    """

    key_bytes = key.encode(encoding=CODING)
    key_len = len(key_bytes)

    enc_bytes = base64.urlsafe_b64decode(msg.encode(encoding=CODING))
    msg_len = len(enc_bytes)
    msg_bytes = bytes(
            (BYTE_CEIL + enc_bytes[i] - key_bytes[i%key_len]) % BYTE_CEIL
            for i in range(msg_len)
        )
    msg = msg_bytes.decode(encoding=CODING)
    return msg
# }}}


if __name__ == "__main__":
    raw_msg = "abcdegahjhjgh"
    enc_msg, key = encrypt(raw_msg)
    dec_msg = decrypt(enc_msg, key)
    assert dec_msg == raw_msg
    print("The key used to encrypt msg: ", key)
    print("The encrypted msg: ", enc_msg)
    print("The decrypted msg: ", dec_msg)
