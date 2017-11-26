from IceNumerics.Spins import *
from IceNumerics.ColloidalIce import ColloidalIce

TargetDir = 'C:\\Users\\aortiza\\Desktop\\2017-06-06-Data\\Spin Ice Sim - Low Res Test - Disorder\\'

print(TargetDir)

Filename = "LAMMPSTest_2017_06_06_16_21_29"

DataFile = open(TargetDir+"Data and Run Files\\"+Filename+'.data','r')

Line = DataFile.readline()
while Line != "Atoms\n":
    Line = DataFile.readline()
    if 'atom types' in Line:
        types = float(Line.split(' ')[0])
    if 'atoms' in Line: 
        atoms = float(Line.split(' ')[0])
    print(Line)

DataFile.readline()

Center = []
Direction = []
Type = 51

while Type == 51:
    Line = DataFile.readline()
    LineFloat = [float(i) for i in Line.replace('\t',' ').split(' ') if i!='']
    Type = LineFloat[1]
    Atom = LineFloat[0]
    Center = Center+[LineFloat[2:5]]
    Direction = Direction+[LineFloat[8:11]]

S = tuple(Spin(tuple(c),tuple(d)) for c, d in zip(Center,Direction))
S = Spins(*S)
S.display()

print(Line)
print(Center)
print(Direction)

DataFile.close()