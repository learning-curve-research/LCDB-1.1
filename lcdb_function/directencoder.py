import io
import numpy as np
import gzip
import base64
import json

class DirectEncoder:

    def __init__(self, precision = None):
        if precision is not None and type(precision) != int:
            raise Exception("Precision must be an int or None!")
        self.precision = precision
        
    def chunks(self, lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    def decode_integer_list(self, encoding_str, chunk_size, expected_length):
        nums = []
        r = int(6 / chunk_size)
        for i, sym in enumerate(encoding_str):
            code = str(ord(sym))
            if i < len(encoding_str) - 1:
                code = code.rjust(6, "0")
            else: # last symbol
                expected_fields = expected_length - len(nums)
                code = code.rjust(r * expected_fields, "0")
            probs = [int(c) for c in CompressingEncoder.chunks(code, r)]
            nums += probs
        return nums

    def labelvector_compress(self, arr):
        # Ensure the array is of type int32
        if arr.dtype != np.uint8:
            raise ValueError("Array must be of np.int32 type")
        
        # Convert the numpy array to bytes
        byte_data = arr.tobytes()
        
        # Compress the byte data using gzip
        compressed_data = gzip.compress(byte_data)
        
        # Encode the compressed data to base64
        base64_encoded_data = base64.b64encode(compressed_data)
        
        # Convert base64 bytes to string
        base64_string = base64_encoded_data.decode('utf-8')
        
        return base64_string

    def labelvector_decompress(self, base64_string):
        # Decode the base64 string to get compressed data
        compressed_data = base64.b64decode(base64_string)
        
        # Decompress the data using gzip
        byte_data = gzip.decompress(compressed_data)
        
        # Convert the byte data back to a numpy array
        arr = np.frombuffer(byte_data, dtype=np.uint8)
        
        return arr

    def encode_label_vector(self, v):
        labels = [str(u) for u in np.unique(v)]
        return labels, [labels.index(str(l)) for l in v]

    def encode_label_vector_compression(self, v):
        labels = [str(u) for u in np.unique(v)]
        label_list = [labels.index(str(l)) for l in v]
        label_list_numpy = np.array(label_list, dtype=np.uint8)
        label_list_compressed = self.labelvector_compress(label_list_numpy)
        return labels, label_list_compressed

    def decode_label_vector(self, descriptor):
        label_names, labels = descriptor[0], descriptor[1]
        return np.array([label_names[i] for i in labels])
    
    def decode_label_vector_decompression(self, descriptor):
        label_names, labels_compressed = descriptor[0], descriptor[1]
        labels = self.labelvector_decompress(labels_compressed)
        return np.array([label_names[i] for i in labels])
        
    def encode_distribution(self, arr):
        if len(arr.shape) == 2:
            encoded = arr[:,:-1].astype("float32")
        else:
            encoded = arr[:].astype("float32")
        if self.precision is not None:
            encoded = np.round(encoded, self.precision)
            if self.precision <= 2:
                encoded = np.round(encoded * (10**self.precision)).astype(int)
        return (self.precision, encoded.tolist())
    
    def proba_compress(self, arr):
        if arr.dtype != np.float16:
            raise ValueError("Array must be of np.float16 type")
        
        # Get the shape of the array
        shape = arr.shape
        
        # Convert the numpy array to bytes
        byte_data = arr.tobytes()
        
        # Compress the byte data using gzip
        compressed_data = gzip.compress(byte_data)
        
        # Encode the compressed data to base64
        base64_encoded_data = base64.b64encode(compressed_data)
        
        # Convert base64 bytes to string
        base64_string = base64_encoded_data.decode('utf-8')
        
        # Create a dictionary to hold the shape and the base64 string
        data_dict = {
            'shape': shape,
            'data': base64_string
        }
        
        return data_dict

    def proba_decompress(self, json_data):
        # Deserialize the JSON to get the shape and base64 string
        shape = tuple(json_data['shape'])
        base64_string = json_data['data']
        
        # Decode the base64 string to get compressed data
        compressed_data = base64.b64decode(base64_string)
        
        # Decompress the data using gzip
        byte_data = gzip.decompress(compressed_data)
        
        # Convert the byte data back to a numpy array
        arr = np.frombuffer(byte_data, dtype=np.float16)
        
        # Reshape the array using the stored shape
        arr = arr.reshape(shape)
        
        return arr
    
    def encode_distribution_compression(self, arr):
        if arr is None:
            return {"shape": None, "data": ""}
        arr = arr.astype("float16")
        arr_compressed = self.proba_compress(arr)
        return arr_compressed

    
    def decode_distribution(self, encoded):
        if encoded is None:
            raise Exception("No (None) distribution given!")
        precision, probs = encoded[0], np.array(encoded[1])
        if precision is not None and precision <= 2:
            probs = probs.astype("float32") / (10**precision)
        probs = np.column_stack([probs, 1 - np.sum(probs, axis=1)])
        
        if precision is not None: # if probs were rounded to a certain precision at encoding time, make sure that the recovered numbers do not encode things that are not there
            probs = np.round(probs, precision)
        return probs
    
    def decode_distribution_decompress(self, encoded):
        arr = self.proba_decompress(encoded)
        return arr

