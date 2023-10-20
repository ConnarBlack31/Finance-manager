import hashlib 
str =  input("Enter the value to encode ")
result = hashlib.sha1(str.encode())
print("The hexadecimal equivalent if SHA1 is :")
print(result.hexdigest())