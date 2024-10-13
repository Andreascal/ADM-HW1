# Say "Hello, World!" With Python
print("Hello, World!")

# Python If-Else
#!/bin/python3
import math
import os
import random
import re
import sys

if __name__ == '__main__':
    n = int(input().strip())
if type(n)== str:
    exit()
else:
    if 1<=n<=100:
        if n % 2 !=0:
            print("Weird")
        else:
            if 2<=n<=5:
                print("Not Weird")
            elif 6<=n<=20:
                print("Weird")
            else:
                print("Not Weird")
    else:
        exit()
                
                
                
                
                

# Arithmetic Operators
if __name__ == '__main__':
    a = int(input())
    b = int(input())

if not 1<=a<=10**10:
    exit()
if not 1<=b<=10**10:
    exit()

print(a+b)
print(a-b)
print(a*b)

    

# Python: Division
if __name__ == '__main__':
    a = int(input())
    b = int(input())
    
print(a//b)
print(a/b)

# Loops
if __name__ == '__main__':
    n = int(input())
if not 1<=n<=20:
    exit()

for i in range(0, n):
    print(i**2)

# Write a function
def is_leap(year):
    if year % 4 == 0:
        if year % 100 ==0:
            if year % 400 == 0:
                leap = True
            else:
                leap= False
        else:
            leap=True
    else:
        leap=False
        
    return leap

# Print Function
if __name__ == '__main__':
    n = int(input())
    
if not 1<=n<=150:
    exit()
    
for s in range(1, n+1):
    print(s, end="")
    

# List Comprehensions
if __name__ == '__main__':
    x = int(input())
    y = int(input())
    z = int(input())
    n = int(input())

rangei=range(0, x+1)
rangej=range(0, y+1)
rangek=range(0, z+1)

matrice=[ [i,j,k] for i in rangei for j in rangej for k in rangek if i+j+k!=n]
print(matrice)

# Find the Runner-Up Score!
if __name__ == '__main__':
    n = int(input())
    arr = map(int, input().split())
rangen=range(2, 10+1)
rangearr=range(-100, 100+1)
lista=sorted(list(arr))
listaset=list(set(lista))
listaset=sorted(listaset, reverse=True)
secondo=listaset[1]
print(secondo)

# Nested Lists
if __name__ == '__main__':
    studenti=[]
    for N in range(int(input())):
        name = input()
        score = float(input())
        studenti.append([name, score])
    
        
listascore=sorted(set(x[1] for x in studenti))
secondo=listascore[1]
listanomi=[x[0] for x in studenti if x[1]==secondo]
listanomi=sorted(listanomi)
i=int()
for i in range(0, len(listanomi)):
    print(listanomi[i])

# Finding the percentage
if __name__ == '__main__':
    n = int(input())
    student_marks = {}
    for _ in range(n):
        name, *line = input().split()
        scores = list(map(float, line))
        student_marks[name] = scores
    query_name = input()
listasco=student_marks[query_name]
score=round(sum(listasco)/len(listasco), 4)
score = "{:.2f}".format(score)
print(score)

# Lists
if __name__ == '__main__':
    # Leggiamo il numero di comandi
    n = int(input())

# Inizializziamo la lista vuota
lista = []

# Iteriamo sui comandi
for _ in range(n):
    # Leggiamo il comando e i suoi parametri
    comando = input().split()
    
    # Identifichiamo il comando e lo eseguiamo
    if comando[0] == "insert":
        # inserisce l'elemento 'e' alla posizione 'i'
        lista.insert(int(comando[1]), int(comando[2]))
    elif comando[0] == "print":
        # stampa la lista
        print(lista)
    elif comando[0] == "remove":
        # rimuove la prima occorrenza di 'e'
        lista.remove(int(comando[1]))
    elif comando[0] == "append":
        # aggiunge 'e' alla fine della lista
        lista.append(int(comando[1]))
    elif comando[0] == "sort":
        # ordina la lista
        lista.sort()
    elif comando[0] == "pop":
        # rimuove l'ultimo elemento della lista
        lista.pop()
    elif comando[0] == "reverse":
        # inverte la lista
        lista.reverse()

# Tuples
if __name__ == '__main__':
    n = int(input())
    integer_list = map(int, input().split())
t=tuple(integer_list)
h=hash(t)
print(h)

# sWAP cASE
def swap_case(s):
    st = ""
    for i in range(0, len(s)):
        if s[i].isupper():
            st+=s[i].lower()
        else:
            st+=s[i].upper()
    return st

# String Split and Join
def split_and_join(line):
    # write your code here
    line= line.split(" ")
    line= "-".join(line)
    return line


# What's Your Name?
#
# Complete the 'print_full_name' function below.
#
# The function is expected to return a STRING.
# The function accepts following parameters:
#  1. STRING first
#  2. STRING last
#
def print_full_name(first, last):
    print( "Hello " + first + " " + last + "! You just delved into python." )


# Mutations
def mutate_string(string, position, character):
    l=list(string)
    l[position]=character
    mutate_string="".join(l)
    return mutate_string

# Find a string

def count_substring(string, sub_string):
    c=0
    j=0
    for i in range(0, (len(string)-len(sub_string)+1)):
        j= len(sub_string)+i
        if sub_string==string[i:j]:
            c+=1
    return c

# String Validators
if __name__ == '__main__':
    s = input()
    print(any(c.isalnum() for c in s))
    print(any(c.isalpha() for c in s))
    if any(c.isdigit() for c in s):
        print("True")
    else:
        print("False")
    if any(c.islower() for c in s):
        print("True")
    else:
        print("False")
    if any(c.isupper() for c in s):
        print("True")
    else:
        print("False")

# Text Alignment
#Replace all ______ with rjust, ljust or center. 
thickness = int(input()) #This must be an odd number
c = 'H'
#Top Cone
for i in range(thickness):
    print((c*i).rjust(thickness-1)+c+(c*i).ljust(thickness-1))
#Top Pillars
for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))
#Middle Belt
for i in range((thickness+1)//2):
    print((c*thickness*5).center(thickness*6))    
#Bottom Pillars
for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))    
#Bottom Cone
for i in range(thickness):
    print(((c*(thickness-i-1)).rjust(thickness)+c+(c*(thickness-i-1)).ljust(thickness)).rjust(thickness*6))

# Text Wrap

def wrap(string, max_width):
    l=[]
    for i in range(0, len(string), max_width):
        l=string[i: (i+max_width)]
        print(l)
        if i+max_width>len(string):
            exit()    
    return l

# Designer Door Mat
# Enter your code here. Read input from STDIN. Print output to STDOUT
# Input dei valori N e M
N, M = map(int, input().split())
for i in range(N//2):
    pattern = ".|." * (2*i + 1)
    print(pattern.center(M, '-'))
print("WELCOME".center(M, '-'))

for i in range(N//2-1, -1, -1):
    pattern = ".|." * (2*i + 1)
    print(pattern.center(M, '-'))

# String Formatting
def print_formatted(number):
    width = len(bin(number))- 2  
    
    for i in range(1, number + 1):
        decimal = str(i)
        octal = oct(i)[2:] 
        hexadecimal = hex(i)[2:].upper()
        binary = bin(i)[2:]  
        
        print(f"{decimal:>{width}} {octal:>{width}} {hexadecimal:>{width}} {binary:>{width}}")

# Alphabet Rangoli
alfabeto = 'abcdefghijklmnopqrstuvwxyz'
def print_rangoli(size):
    row=[]
    for i in range(0, size):
       stampa= '-'.join(alfabeto[i:size])
       row.append(stampa[::-1]+stampa[1:])
    width=len(row[0])
    
    for i in range(size-1, 0, -1):
        print(row[i].center(width,'-'))
    
    for i in range(0, size):
        print(row[i].center(width, '-'))

# Capitalize!

# Complete the solve function below.
def solve(s):
    c=""
    for i in range(len(s)):
        if s[i].isalpha():
            if i==0:
                c+=s[i].upper()
            elif s[i-1]==" ":
                c+=s[i].upper()
            else:
                c+=s[i].lower()
        else:
            c+=s[i]
    return c

# The Minion Game
def minion_game(string):
    voc="AEIOU"
    Stuart=0
    Kevin=0
    for i in range(len(string)):
        if string[i] in voc:
            Kevin+=(len(string)-i)
        else:
            Stuart+=(len(string)-i)
    if Stuart>Kevin:
        print("Stuart", Stuart)
    elif Kevin>Stuart:
        print("Kevin", Kevin)
    else:
        print("Draw")



# Introduction to Sets
def average(array):
    s=set(array)
    ar= list(s)
    l=0
    for i in range(len(ar)):
        l+=int(ar[i])
    l= l/len(ar)
    return l
    

# Merge the Tools!
def merge_the_tools(string, k):
    for i in range(0, len(string), k):
        l=string[i: i+k]
        m=""
        s=set()
        for j in l:
            if j not in s:
                m+=j
                s.add(j)
        print(m)


# No Idea!
# Enter your code here. Read input from STDIN. Print output to STDOUT
if __name__ == '__main__':
    n, m = (int, input().split())
    array = list(map(int, input().split()))
    A = set(map(int, input().split()))
    B = set(map(int, input().split()))
    h=0
    
    for i in range(len(array)):
        if array[i] in A:
            h+=1
        elif array[i] in B:
            h-=1
    print(h)


# Symmetric Difference
# Enter your code here. Read input from STDIN. Print output to STDOUT
m = int(input())
a = set(map(int, input().split()))
n = int(input())
b = set(map(int, input().split()))
symmetric_diff = a.symmetric_difference(b)
ris = sorted(symmetric_diff)
for num in ris:
    print(num)

# Set .add()
# Enter your code here. Read input from STDIN. Print output to STDOUT
N=int(input())
paesi=[]
for _ in range(N):
    p=input()
    paesi.append(p)
s=len(set(paesi))
print(s)

# Set .discard(), .remove() & .pop()
n = int(input())
s = set(map(int, input().split()))
m = int(input())
for _ in range(m):
    command = input().split()
    if command[0] == "pop":
        s.pop()
    elif command[0] == "remove":
        s.remove(int(command[1]))
    elif command[0] == "discard":
        s.discard(int(command[1]))
        
        
print(sum(s))

# Set .union() Operation
# Enter your code here. Read input from STDIN. Print output to STDOUT
n=int(input())
N=set(map(int, input().split()))
b=int(input())
B=set(map(int, input().split()))
print(len(N.union(B)))

# Set .intersection() Operation
# Enter your code here. Read input from STDIN. Print output to STDOUT
n=int(input())
N=set(map(int, input().split()))
b=int(input())
B=set(map(int, input().split()))
print(len(N.intersection(B)))

# Set .difference() Operation
# Enter your code here. Read input from STDIN. Print output to STDOUT
n=int(input())
N=set(map(int, input().split()))
b=int(input())
B=set(map(int, input().split()))
print(len(N-B))

# Set .symmetric_difference() Operation
# Enter your code here. Read input from STDIN. Print output to STDOUT
n=int(input())
N=set(map(int, input().split()))
b=int(input())
B=set(map(int, input().split()))
print(len(N.symmetric_difference(B)))

# Set Mutations
# Enter your code here. Read input from STDIN. Print output to STDOUT
a=int(input())
A=set(map(int, input().split()))
N=int(input())
for _ in range(N):
    command=input().split()
    lista=set(map(int, input().split()))
    if command[0]=="intersection_update":
        A.intersection_update(lista)
    elif command[0]=="update":
        A.update(lista)
    elif command[0]=="difference_update":
        A.difference_update(lista)
    elif command[0]=="symmetric_difference_update":
        A.symmetric_difference_update(lista)
print(sum(A))

# The Captain's Room
# Enter your code here. Read input from STDIN. Print output to STDOUT
K=int(input())
if not 1<K<1000:
    exit()
lista=list(map(int, input().split()))
conteggio = {}
for numero in lista:
    if numero in conteggio:
        conteggio[numero] += 1
    else:
        conteggio[numero] = 1
for i, l in conteggio.items():
    if l!=K:
        print(i)
        

# Check Subset
# Enter your code here. Read input from STDIN. Print output to STDOUT
T=int(input())
for _ in range(T):
    a=int(input())
    A=set(map(int, input().split()))
    b=int(input())
    B=set(map(int, input().split()))
    if A.intersection(B)==A:
        print("True")
    else:
        print("False")
    

# Check Strict Superset
# Enter your code here. Read input from STDIN. Print output to STDOUT
A = set(map(int, input().split()))
n = int(input())
C=[]
for _ in range(n):
    B=set(map(int, input().split()))
    if B.intersection(A)==B:
        if A.union(B)==A:
            if len(A)>len(B):
                C.append("True")
    else:
        print("False")
        exit()
print("True")

# collections.Counter()
# Enter your code here. Read input from STDIN. Print output to STDOUT
X=int(input())
x=list(map(int, input().split()))
N=int(input())
earn=0
for _ in range(N):
    size, price = map(int, input().split())
    if size in x:
        earn += price
        x.remove(size)
print(earn)

# DefaultDict Tutorial
# Enter your code here. Read input from STDIN. Print output to STDOUT
from collections import defaultdict
n, m=map(int, input().split())
diz=defaultdict(list)
for i in range(1, n+1):
    word = input().strip()
    diz[word].append(i)
for _ in range(m):
    word_b = input().strip()
    if word_b in diz:
        print(' '.join(map(str, diz[word_b])))
    else:
        print(-1)

# Collections.namedtuple()
# Enter your code here. Read input from STDIN. Print output to STDOUT
from collections import namedtuple
# Legge il numero totale di studenti
N = int(input().strip())
columns = input().strip().split()
Student = namedtuple('Student', columns)
students = []
voti=[]
for _ in range(N):
    data = input().strip().split()
    student = Student(*data)
    students.append(student)
for student in students:
    voti.append(int(student.MARKS))
print("{:.2f}".format(sum(voti)/len(voti)))
#for j in range(N):
   # if students


# itertools.product()
# Enter your code here. Read input from STDIN. Print output to STDOUT
from itertools import product
A=list(map(int, input().split()))
B=list(map(int, input().split()))
print(" ".join(map(str, product(A,B))))

# Collections.OrderedDict()
# Enter your code here. Read input from STDIN. Print output to STDOUT
from collections import OrderedDict
from collections import Counter
N=int(input())
diz={}
dizionario=OrderedDict()
for _ in range(N):
    prodotto, prezzo=input().rsplit(' ', 1)
    prezzo=int(prezzo)
    if prodotto in dizionario:
        dizionario[prodotto] += prezzo
    else:
        dizionario[prodotto] = prezzo
for i, j in dizionario.items():
    print(i, j)



































































































# Word Order
# Enter your code here. Read input from STDIN. Print output to STDOUT
from collections import OrderedDict
n=int(input())
l = OrderedDict()
for _ in range(n):
    riga=str(input())
    if riga in l:
        l[riga]+=1
    else:
        l[riga]=1
print(len(l))
print(" ".join(map(str, l.values())))

# Collections.deque()
# Enter your code here. Read input from STDIN. Print output to STDOUT
from collections import deque
n=int(input())
d=deque()
for _ in range(n):
    dati = input().split()
    if len(dati)>1:
        comando=dati[0]
        numero=int(dati[1])
    elif len(dati)==1:
        comando=dati[0]
    if comando=="append":
        d.append(numero)
    if comando=="appendleft":
        d.appendleft(numero)
    if comando=="clear":
        d.clear()
    if comando=="pop":
        d.pop()
    if comando=="popleft":
        d.popleft()
    if comando=="extend":
        d.extend(numero)
    if comando=="extendleft":
        d.extendleft(numero)
    if comando=="remove":
        d.remove(numero)
print(" ".join(map(str, d)))


# Company Logo
import math
import os
import random
import re
import sys
def carattericomuni(s):
    diz = {}
    for c in s:
        if c in diz:
            diz[c] += 1
        else:
            diz[c] = 1
    lista = [(c, count) for c, count in diz.items()]
    for i in range(len(lista)):
        for j in range(i + 1, len(lista)):
            if lista[i][0] > lista[j][0]:
                lista[i], lista[j] = lista[j], lista[i]
                
    for i in range(len(lista)):
        for j in range(i + 1, len(lista)):
            if lista[i][1] < lista[j][1]:
                lista[i], lista[j] = lista[j], lista[i]
            elif lista[i][1] == lista[j][1] and lista[i][0] > lista[j][0]:
                lista[i], lista[j] = lista[j], lista[i]
                
    for i in range(3):
        print(lista[i][0], lista[i][1])
if __name__ == "__main__":
    s = input().strip()
    carattericomuni(s)

# Piling Up!
# Enter your code here. Read input from STDIN. Print output to STDOUT
T=int(input())
for _ in range(T):
    n=int(input())
    lista=list(map(int, input().split()))
    ultimo=float("Inf")
    sinistra=0
    destra=n-1
    possibile="True"
    
    while sinistra<=destra:
        if lista[sinistra]>=lista[destra]:
            numero=lista[sinistra]
            sinistra+=1
        else:
            numero=lista[destra]
            destra-=1
    
        if numero>ultimo:
            possibile="False"
            break
    
        ultimo=numero
    if possibile=="True":
        print("Yes")
    else:
        print("No")
        
        
        
        
        

# Calendar Module
# Enter your code here. Read input from STDIN. Print output to STDOUT
import calendar
month, day, year = map(int, input().split())
weekday_number = calendar.weekday(year, month, day)
days = ['MONDAY', 'TUESDAY', 'WEDNESDAY', 'THURSDAY', 'FRIDAY', 'SATURDAY', 'SUNDAY']
print(days[weekday_number])

# Time Delta
#!/bin/python3
import math
import os
import random
import re
import sys
from datetime import datetime
# Complete the time_delta function below.
def time_delta(t1, t2):
    time_format = "%a %d %b %Y %H:%M:%S %z"
    dt1 = datetime.strptime(t1, time_format)
    dt2 = datetime.strptime(t2, time_format)
    delta = abs(int((dt1 - dt2).total_seconds()))
    return delta
    
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    t = int(input().strip())
    for t_itr in range(t):
        t1 = input().strip()
        t2 = input().strip()
        delta = time_delta(t1, t2)
        fptr.write(str(delta) + '\n')
    fptr.close()


# Exceptions
# Enter your code here. Read input from STDIN. Print output to STDOUT
T=int(input())
for _ in range(T):
    a, b=input().split()
    try:
        print(int(a)//int(b))
    except ZeroDivisionError as e:
        print("Error Code: integer division or modulo by zero")
    except ValueError as e:
        print("Error Code:", e)

# Zipped!
# Enter your code here. Read input from STDIN. Print output to STDOUT
N, X=map(int, input().split())
voti = [list(map(float, input().split())) for _ in range(X)]

for studente in zip(*voti):
    print("{:.1f}".format(sum(studente) / X))


# Athlete Sort
#!/bin/python3
import math
import os
import random
import re
import sys

if __name__ == '__main__':
    nm = input().split()
    n = int(nm[0])
    m = int(nm[1])
    arr = []
    for _ in range(n):
        arr.append(list(map(int, input().rstrip().split())))
    k = int(input())
    arr.sort(key=lambda x: x[k])
    for i in arr:
        print(" ".join(map(str, i)))

# ginortS
# Enter your code here. Read input from STDIN. Print output to STDOUT
def sorting(s):
    minus = sorted([c for c in s if c.islower()])
    maius = sorted([c for c in s if c.isupper()])
    dispari = sorted([c for c in s if c.isdigit() and int(c) % 2 == 1])
    pari = sorted([c for c in s if c.isdigit() and int(c) % 2 == 0])
    return ''.join(minus + maius + dispari + pari)
if __name__ == '__main__':
    s = input()
    print(sorting(s))

# Map and Lambda Function
cube = lambda x: x**3  # complete the lambda function 
lista=[]
def fibonacci(n):
    if n>=1:
        lista.append(0)
    if n>=2:
        lista.append(1)
    if n>=3:
        for i in range(2, n):
            lista.append(lista[-1]+lista[-2])
    return lista


# Detect Floating Point Number
# Enter your code here. Read input from STDIN. Print output to STDOUT
def f(s):
    try:
        float(s)
        return s.count('.') == 1 and any(c.isdigit() for c in s if c != '.')
    except ValueError:
        return False
if __name__ == '__main__':
    n = int(input())
    for _ in range(n):
        s = input()
        print(f(s))
        
        
        
        

# Re.split()
regex_pattern = r"[,.]"	# Do not delete 'r'.

# Group(), Groups() & Groupdict()
# Enter your code here. Read input from STDIN. Print output to STDOUT
import re
stringa = input()
pattern = r'([a-zA-Z0-9])\1'
ricerca = re.search(pattern, stringa)
if ricerca:
    print(ricerca.group(1))
else:
    print(-1)


# Re.findall() & Re.finditer()
# Enter your code here. Read input from STDIN. Print output to STDOUT
import re
s = input()
pattern = r'(?<=[^aeiouAEIOU])([aeiouAEIOU]{2,})(?=[^aeiouAEIOU])'
trova = re.findall(pattern, s)
if trova:
    for i in trova:
        print(i)
else:
    print(-1)

# Re.start() & Re.end()
# Enter your code here. Read input from STDIN. Print output to STDOUT
import re
s = input()
k = input()
trova = list(re.finditer(r'(?={})'.format(re.escape(k)), s))
# Se troviamo corrispondenze, stampiamo gli indici di inizio e fine
if trova:
    for i in trova:
        start = i.start()
        print((start, start + len(k) - 1))
else:
    print((-1, -1))

# Regex Substitution
# Enter your code here. Read input from STDIN. Print output to STDOUT
import re
def mod(text):
    text = re.sub(r'(?<= )&&(?= )', 'and', text) 
    text = re.sub(r'(?<= )\|\|(?= )', 'or', text)
    return text
if __name__ == '__main__':
    n = int(input())
    for _ in range(n):
        r = input()
        modifica= mod(r)
        print(modifica)

# XML 1 - Find the Score

def get_attr_number(node):
    lun= len(node.attrib)
    for i in node:
        lun += get_attr_number(i)
    
    return lun

# XML2 - Find the Maximum Depth

maxdepth = 0
def depth(elem, level):
    global maxdepth
    maxdepth = max(maxdepth, level + 1)
    for child in elem:
        depth(child, level + 1)

# Standardize Mobile Number Using Decorators
def wrapper(f):
    def fun(l):
        standardized_numbers = ['+91 {} {}'.format(num[-10:-5], num[-5:]) for num in l]
        return f(standardized_numbers)
    return fun

# Decorators 2 - Name Directory

def person_lister(f):
    def inner(people):
        people.sort(key=lambda x: int(x[2]))
        return [f(person) for person in people]
    return inner

# Arrays

def arrays(arr):
    l=[]
    for i in range(1, len(arr)+1):
        l.append(float(arr[-i]))
    a=numpy.array(l)
    return a

# Shape and Reshape
import numpy as np
array=list(map(int, input().split()))
arr=np.array(array)
finale=np.reshape(arr, (3, 3))
print(finale)

# Transpose and Flatten
import numpy as np
N, M = map(int, input().split())
mat=[]
for i in range(N):
    mat.append(input().split())
    for j in range(len(mat[i])):
        mat[i][j]=int(mat[i][j])
mat=np.array(mat)
print(np.transpose(mat))
flat=mat.flatten()
print(flat)


# Concatenate
import numpy as np
N, M, P=map(int, input().split())
l=[]
for i in range(N):
    l.append(list(map(int, input().split())))
for i in range(M):
    l.append(list(map(int, input().split())))
print(np.array(l))

# Zeros and Ones
import numpy as np

n= list(map(int, input().split()))
print(np.zeros(tuple(n), dtype=np.int64))
print(np.ones(tuple(n), dtype=np.int64))

# Eye and Identity
import numpy as np
np.set_printoptions(legacy='1.13')
N, M=map(int, input().split())
print(np.eye(N, M))

# Array Mathematics
import numpy as np
X, Y = map(int, input().split())
A = np.array([list(map(int, input().split())) for _ in range(X)])
B = np.array([list(map(int, input().split())) for _ in range(X)])
print(A+B)
print(A-B)
print(A*B)
print(A//B)
print(A % B)
print(A**B)

# Floor, Ceil and Rint
import numpy as np
np.set_printoptions(legacy='1.13')
riga=list(map(float, input().split()))
r=np.array(riga)
print(np.floor(r))
print(np.ceil(r))
print(np.rint(r))

# Sum and Prod
import numpy as np
X, Y = map(int, input().split())
a = np.array([list(map(int, input().split())) for _ in range(X)])
somma = np.sum(a, axis=0)
r = np.prod(somma)
print(r)



# Min and Max
import numpy as np
X, Y = map(int, input().split())
a= np.array([list(map(int, input().split())) for _ in range(X)])
m = np.min(a, axis=1)
massimo = np.max(m)
print(massimo)

# Mean, Var, and Std
import numpy as np
X, Y = map(int, input().split())
a = np.array([list(map(int, input().split())) for _ in range(X)])
print(np.mean(a, axis=1))
print(np.var(a, axis=0))
if np.std(a)==0:
    print(0.0)
else:
    print(f"{np.std(a):.11f}")

# Validating Roman Numerals
regex_pattern = r"^M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$" 	# Do not delete 'r'.

# Validating phone numbers
# Enter your code here. Read input from STDIN. Print output to STDOUT
import re
def numerovalido(num):
    pattern= r"^[789]\d{9}$"
    return bool(re.match(pattern, num))
n= int(input())
for _ in range(n):
    riga= input().strip()
    if numerovalido(riga):
        print("YES")
    else:
        print("NO")

# Dot and Cross
import numpy as np
N=int(input())
A = np.array([list(map(int, input().split())) for _ in range(N)])
B = np.array([list(map(int, input().split())) for _ in range(N)])
print(np.dot(A,B))

# Inner and Outer
import numpy as np
A = np.array([list(map(int, input().split()))])
B = np.array([list(map(int, input().split()))])
print(np.inner(A, B)[0][0])
print(np.outer(A, B))

# Polynomials
import numpy as np
P = list(map(float, input().split()))
x = float(input())
ris = np.polyval(P, x)
print(ris)



# Linear Algebra
import numpy as np
n = int(input())
m = []
for _ in range(n):
    riga = list(map(float, input().split()))
    m.append(riga)
m = np.array(m)
print(round(np.linalg.det(m), 2))



# Validating and Parsing Email Addresses
import re
import email.utils
n = int(input().strip())
pattern = r'^[a-zA-Z][\w._-]*@[a-zA-Z]+\.[a-zA-Z]{1,3}$'
for _ in range(n):
    riga = input().strip()
    nome, mail = email.utils.parseaddr(riga)
    if re.match(pattern, mail):
        print(f"{nome} <{mail}>")



# HTML Parser - Part 1
# Enter your code here. Read input from STDIN. Print output to STDOUT
from html.parser import HTMLParser

N = int(input())
riga = [input() for _ in range(N)]

class MyHTMLParser(HTMLParser):
    def handle_starttag(self, tag, att):
        print("Start :", tag)
        for i in att:
            print(f"-> {i[0]} > {i[1] if i[1] is not None else 'None'}")

    def handle_endtag(self, tag):
        print("End   :", tag)

    def handle_startendtag(self, tag, att):
        print("Empty :", tag)
        for j in att:
            print(f"-> {j[0]} > {j[1] if j[1] is not None else 'None'}")

parser = MyHTMLParser()
for l in riga:
    parser.feed(l)


# HTML Parser - Part 2
from html.parser import HTMLParser
class MyHTMLParser(HTMLParser):
    def handle_comment(self, data):
        if '\n' in data:
            print(">>> Multi-line Comment")
            print(data)
        else:
            print(">>> Single-line Comment")
            print(data)
    
    def handle_data(self, data):
        if data != '\n':
            print(">>> Data")
            print(data)
html = ""
for i in range(int(input())):
    html += input().rstrip()
    html += '\n'
parser = MyHTMLParser()
parser.feed(html)




# Validating UID
# Enter your code here. Read input from STDIN. Print output to STDOUT
import re
n = int(input())
for i in range(n):
    UID = input()
    a = len(re.findall(r"[A-Z]", UID)) >= 2
    b = len(re.findall(r"[0-9]", UID)) >= 3
    c = bool(re.match(r"^[a-zA-Z0-9]+$", UID))
    d = len(set(UID)) == len(UID)
    e = len(UID) == 10
    
    
    if all([a, b, c, d, e]):
        print("Valid")
    else:
        print("Invalid")

# Validating Credit Card Numbers
# Enter your code here. Read input from STDIN. Print output to STDOUT
import re
pattern = r"^(4|5|6)\d{3}(-?\d{4}){3}$"
no_repeated_digits = r"(?!.*(\d)(-?\1){3})"
n = int(input())
for i in range(n):
    card = input()
    i = len(re.findall(r"[0-9]", card)) == 16
    j = bool(re.match(pattern, card))
    k = bool(re.match(no_repeated_digits, card))
    
    if all([i, j, k]):
        print("Valid")
    else:
        print("Invalid")





# Birthday Cake Candles
#!/bin/python3
import math
import os
import random
import re
import sys

def birthdayCakeCandles(candles):
    massimo = max(candles)
    conteggio = candles.count(massimo)
    return conteggio
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    candles_count = int(input().strip())
    candles = list(map(int, input().rstrip().split()))
    result = birthdayCakeCandles(candles)
    fptr.write(str(result) + '\n')
    fptr.close()

# Number Line Jumps
#!/bin/python3
import math
import os
import random
import re
import sys
#
# Complete the 'kangaroo' function below.
#
# The function is expected to return a STRING.
# The function accepts following parameters:
#  1. INTEGER x1
#  2. INTEGER v1
#  3. INTEGER x2
#  4. INTEGER v2
#
def kangaroo(x1, v1, x2, v2):
    if v1 == v2:
        return "YES" if x1 == x2 else "NO"
    pos = x2 - x1
    vel = v1 - v2
    if pos % vel == 0 and pos / vel >= 0:
        return "YES"
    return "NO"
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    first_multiple_input = input().rstrip().split()
    x1 = int(first_multiple_input[0])
    v1 = int(first_multiple_input[1])
    x2 = int(first_multiple_input[2])
    v2 = int(first_multiple_input[3])
    result = kangaroo(x1, v1, x2, v2)
    fptr.write(result + '\n')
    fptr.close()

# Viral Advertising
#!/bin/python3
import math
import os
import random
import re
import sys
#
# Complete the 'viralAdvertising' function below.
#
# The function is expected to return an INTEGER.
# The function accepts INTEGER n as parameter.
#
def viralAdvertising(n):
    shared = 5
    l = 0
    
    for day in range(1, n + 1):
        liked = shared // 2
        l += liked
        shared = liked * 3
    return l
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    n = int(input().strip())
    result = viralAdvertising(n)
    fptr.write(str(result) + '\n')
    fptr.close()

# Recursive Digit Sum
#!/bin/python3
import math
import os
import random
import re
import sys
#
# Complete the 'superDigit' function below.
#
# The function is expected to return an INTEGER.
# The function accepts following parameters:
#  1. STRING n
#  2. INTEGER k
#
def superDigit(n, k):
    i = sum(int(digit) for digit in n) * k
    
    def superdigit(v):
        if v < 10:
            return v
        else:
            w = sum(int(digit) for digit in str(v))
            return superdigit(w)
    return superdigit(i)
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    first_multiple_input = input().rstrip().split()
    n = first_multiple_input[0]
    k = int(first_multiple_input[1])
    result = superDigit(n, k)
    fptr.write(str(result) + '\n')
    fptr.close()

# Insertion Sort - Part 1
#!/bin/python3
import math
import os
import random
import re
import sys
#
# Complete the 'insertionSort1' function below.
#
# The function accepts following parameters:
#  1. INTEGER n
#  2. INTEGER_ARRAY arr
#
def insertionSort1(n, arr):
    arr_1 = arr[-1]
    i = n - 2
    while i >= 0 and arr[i] > arr_1:
        arr[i + 1] = arr[i]
        print(" ".join(map(str, arr)))
        i -= 1
    arr[i + 1] = arr_1
    print(" ".join(map(str, arr)))
if __name__ == '__main__':
    n = int(input().strip())
    
    arr = list(map(int, input().rstrip().split()))
    insertionSort1(n, arr)

# Insertion Sort - Part 2
#!/bin/python3
import math
import os
import random
import re
import sys
#
# Complete the 'insertionSort2' function below.
#
# The function accepts following parameters:
#  1. INTEGER n
#  2. INTEGER_ARRAY arr
#
def insertionSort2(n, arr):
    for i in range(1, n):
        a = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > a:
            arr[j + 1] = arr[j]
            j -= 1
        
        # Insert the current element at the correct position
        arr[j + 1] = a
        
        # Print the current state of the array after each insertion
        print(" ".join(map(str, arr)))

if __name__ == '__main__':
    n = int(input().strip())
    arr = list(map(int, input().rstrip().split()))
    insertionSort2(n, arr)

# Hex Color Code
# Enter your code here. Read input from STDIN. Print output to STDOUT
import re
def funzione(linee):
    pattern = r'#[0-9A-Fa-f]{6}|#[0-9A-Fa-f]{3}'
    lista = []
    a = False
    
    for i in linee:
        if '{' in i:
            a = True
        if '}' in i:
            a = False
        
        if a:
            cerca = re.findall(pattern, i)
            lista.extend(cerca)
    
    return lista
if __name__ == '__main__':
    n = int(input().strip())
    linee = [input().strip() for _ in range(n)]
    
    colori = funzione(linee)
    
    for i in colori:
        print(i)

