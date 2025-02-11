from test import TextToNumber

ob=TextToNum("coding is super fun, but hard to learn !")
ob.cleaner()
ob.token()
ob.removestop()
dt=ob.stemme()
print(dt)