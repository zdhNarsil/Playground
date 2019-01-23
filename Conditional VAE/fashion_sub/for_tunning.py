import os

if __name__ == '__main__':
    for c_2 in [0.1, 0.3, 0.5, 0.7, 0.9]:
        for c_e in [0.1, 0.3, 0.5, 0.7, 0.9]:
            os.system('python cvaeVATz.py --epoch 35 --coef-vat2 ' + str(c_2)
                  + ' --coef-ent ' + str(c_e))