""" used in sample latent stats """
from multiprocessing import Pool
def return_id_and_records(process_id, records):
    """ for testing """
    return process_id, records


def test_starmap():
    chunks = [
        ['B', 'b'], ['a', 'A']
    ]

    with Pool(2) as p:
        results = p.starmap(
            return_id_and_records,
            zip(range(2), chunks)
        )

    for res in results:

        if res[0] == 0:
            assert res[1] == ['B', 'b']
        else:
            assert res[1] == ['a', 'A']




