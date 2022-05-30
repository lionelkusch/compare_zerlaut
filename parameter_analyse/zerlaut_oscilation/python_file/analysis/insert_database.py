import sqlite3
import numpy as np
import sys


def type_database(variable):
    if hasattr(variable, 'dtype'):
        if np.issubdtype(variable, int):
            return 'INTEGER'
        elif np.issubdtype(variable, float):
            return 'REAL'
        else:
            sys.stderr.write('ERROR bad type of save variable\n')
            exit(1)
    else:
        if isinstance(variable, int):
            return 'INTEGER'
        elif isinstance(variable, float):
            return 'REAL'
        elif isinstance(variable, str):
            return 'TEXT'
        else:
            sys.stderr.write('ERROR bad type of save variable\n')
            exit(1)

def init_database(data_base, table_name):
    """
    Initialise the connection to the database et create the table
    :param data_base: file where is the database
    :param table_name: the name of the table
    :return: the connexion to the database
    """
    variable = ''
    key_variable = ',amplitude,frequency,noise,'
    measures_name = ['path_file', 'names_population', 'frequency_dom', 'PLV_value', 'PLV_angle', 'max_rates',
                     'min_rates', 'mean_rates', 'std_rates', 'amplitude', 'frequency', 'noise']
    measures_name.remove('names_population')
    measures_name.remove('path_file')
    measures = ''
    for name in measures_name:
        measures += name + ' REAL,'

    con = sqlite3.connect(data_base, detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES, timeout=10000)
    cur = con.cursor()
    cur.execute('CREATE TABLE IF NOT EXISTS '
                + table_name
                + '(date TIMESTAMP NOT NULL,'
                  'path_file TEXT NOT NULL,'
                + variable
                + 'names_population TEXT NOT NULL,'
                + measures
                + 'PRIMARY KEY'
                  '(path_file' + key_variable + 'names_population))'
                )
    cur.close()
    con.close()

def check_already_analyse_database(data_base, table_name, result_path, name_population):
    """
    Check if the analysis was already perform
    :param data_base: path of the database
    :param table_name: name of the table
    :param result_path: folder to analyse
    """
    con = sqlite3.connect(data_base, detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES)
    cursor = con.cursor()
    cursor.execute("SELECT * FROM "+table_name+" WHERE path_file = '"+result_path+"' AND names_population='"+name_population+"'")
    check = len(cursor.fetchall()) != 0
    cursor.close()
    con.close()
    return check


def insert_database(data_base, table_name, results):
    """
    Insert some result in the database
    :param data_base: name of database
    :param table_name: the table where insert the value
    :return: nothing
    """
    con = sqlite3.connect(data_base, detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES, timeout=1000)
    cur = con.cursor()
    list_data = []
    for result in results:
        list_data.append(tuple(result.values()))
    keys = ','.join(results[0].keys())
    question_marks = ','.join(list('?' * len(results[0].keys())))
    cur.executemany('INSERT INTO ' + table_name + ' (' + keys + ') VALUES (' + question_marks + ')', list_data)
    con.commit()
    cur.close()
    con.close()


if __name__ == '__main__':
    from parameter_analyse.zerlaut_oscilation.python_file.analysis.analysis import analysis
    path_root = '/home/kusch/Documents/project/Zerlaut/compare_zerlaut/parameter_analyse/zerlaut_oscilation/simulation/'
    database = path_root + "/database_2.db"
    table_name = "exploration"
    init_database(database, table_name)
    for noise in np.arange(1e-9, 1e-8, 1e-9):
    # for noise in np.arange(1e-8, 1e-7, 1e-8):
    # for noise in np.arange(0.0, 1e-5, 5e-7):
        for frequency in np.concatenate(([1], np.arange(5., 51., 5.))):
            path_simulation = path_root + "frequency_" + str(frequency) + "_noise_" + str(noise) + "/"
            if not check_already_analyse_database(database, table_name, path_simulation, 'excitatory'):
                results = analysis(path_simulation)
                insert_database(database, table_name, results)

    