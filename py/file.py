import sys
import pandas as pd


# def parse_result(result, line_index=-2, result_indices=[-3, -1]):
#     print('parse_result')
#     print(result)
#     line = result.split("\n")[line_index].split(' ')
#     line = [w for w in line if w not in ['', '\t', '\t\t']]
#     return (float(line[i]) for i in result_indices)


def get_arg(name: str, default_value=None, flag=False, parse_func=int):
    """ Parse command line args
    """
    try:
        index = sys.argv.index(name)
        value = sys.argv[index + 1]
    except ValueError:
        if default_value is None:
            raise NameError("Unable to find argument: {}".format(name))
        else:
            value = default_value

    except IndexError:
        if flag:
            value = True
        else:
            raise IndexError("No value found for argument: {}".format(name))

    return parse_func(value)


def save_as_latex_table(results: dict, translate: dict, use_all_keys=True,
                        filename='../../result_tables.tex',
                        combine_keys=[('mean', 'std')],
                        param_cols=['n', 'p']):
    """
    results = dict of dataframes
    """
    with open(filename, "w") as f:
        # clear file
        f.write('')

    param_cols_ = param_cols.copy()
    for k, result in results.items():
        assert result is not None
        if combine_keys:
            combine_columns(result, combine_keys)

        # rm unused params
        param_cols = [c for c in param_cols_ if c in result.keys()]
        # sort & reorder s.t. param_cols are shown first
        result.sort_values(by=param_cols, inplace=True)
        print('params cols', param_cols +
              [k for k in result.keys() if k not in param_cols])
        result = result.loc[:, param_cols +
                            [k for k in result.keys() if k not in param_cols]]

        with open(filename, "a") as f:
            print(f'% {k}', file=f)
            # Table head
            header = ' & '.join([translate[c] for c in param_cols])
            if use_all_keys:
                cols = result.keys()
                rest = [c for c in cols if c not in param_cols]
                if rest:
                    header += ' & '
                    header += ' & '.join(rest)
            else:
                cols = param_cols

            header += r' \\'

            n_cols = len(cols)
            print("\n\\begin{table}[] \n\\begin{tabular}{%s}" %
                  (n_cols * 'l'), file=f)
            print(header, file=f)

            # Table body
            for i, row in result.iterrows():
                values = [parse_table_element(v, translate) for v in row[cols]]
                # if row['priority'] == 'Yes':
                #     # don't print duplicate values
                #     for i in range(len(param_cols) - 1):
                #         values[i] = ' '

                print(' \t& '.join(values) + r' \\', file=f)

            print("""\\end{tabular}
    \\caption{\\label{tab:result-%s} %s}
\\end{table}"""
                  % (k, k), file=f)


def parse_table_element(value, translate={}) -> str:
    if value in translate.keys():
        value = translate[value]

    if pd.isna(value):
        return ''

    if isinstance(value, str):
        return value

    if isinstance(value, float):
        if value == 1.0:
            return '1'
        if value == int(value):
            return str(int(value))

        # If possible, write as fraction `rate = 1/mean`
        try:
            a, b = value.as_integer_ratio()
            if a == 1 and b < 100:
                return '%i/%i' % (a, b)
        except OverflowError:
            pass
        except ValueError:
            pass

    return str(round(value, 4))


def combine_columns(data: pd.DataFrame, p,
                    format_func=lambda a, b: f'{a} ({b})'):
    """ Combine columns such as 'x mean', 'x std' into a format such as
    'x: `mean (std)`'

    :params:
    p = list of tuples with paired keys, e.g. [('mean','std')]
    """
    old_keys = []
    for key in data.keys():
        for k1, k2 in p:
            if k1 in key:
                # assume format is either 'x mean' or 'mean x'
                other_key = key.replace(k1, k2)
                # TODO use regex
                prefix = ' '.join(
                    (k for k in key.split(' ') if k not in [k1, k2]))
                n_decimals = 5
                pairs = zip(data[key].round(n_decimals),
                            data[other_key].round(n_decimals))
                formatted = [format_func(a, b) for a, b in pairs]
                data.loc[:, prefix] = pd.Series(formatted, index=data.index)
                old_keys.append(key)
                if f'{prefix} {k2}' in data.keys():
                    old_keys.append(f'{prefix} {k2}')
                elif f'{k2} {prefix}' in data.keys():
                    old_keys.append(f'{k2} {prefix}')

    data.drop(columns=old_keys, inplace=True)
