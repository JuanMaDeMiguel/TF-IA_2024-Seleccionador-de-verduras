import os
from vista import Vista

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    vista = Vista()
    vista.mostrar_menu()








