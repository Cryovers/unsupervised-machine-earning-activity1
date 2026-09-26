import unittest
import numpy as np
import pandas as pd
from atividade1 import preparar, perfilar


class PreparacaoTest(unittest.TestCase):
    def base(self):
        return pd.DataFrame([dict(CustomerID=str(i), InvoiceNo=f'{i}A', InvoiceDate='2011-12-01', Quantity=2, UnitPrice=10, StockCode='P1') for i in range(10)])

    def test_rfm_e_auditoria(self):
        df = self.base()
        original = df.iloc[0].to_dict()
        extras = [original, dict(original, StockCode='P2', Quantity=3),
                  dict(original, InvoiceNo='B', InvoiceDate='2011-12-09'),
                  dict(original, CustomerID=None), dict(original, InvoiceNo='C123'),
                  dict(original, Quantity=-1), dict(original, UnitPrice=0),
                  dict(original, InvoiceDate='invalida')]
        rfm, X, audit = preparar(pd.concat([df, pd.DataFrame(extras)], ignore_index=True))
        self.assertEqual(rfm.loc['0','Frequency'], 2)
        self.assertEqual(rfm.loc['0','Monetary'], 70)
        self.assertEqual(rfm.loc['0','Recency'], 1)
        self.assertEqual(rfm.loc['1','Recency'], 9)
        self.assertEqual(audit['linhas_validas'], 12)
        for key in ['duplicatas','sem_cliente','cancelamentos','quantidade_nao_positiva','preco_nao_positivo','campos_invalidos']:
            self.assertEqual(audit[key], 1, key)
        self.assertAlmostEqual(rfm.Monetary.sum(), audit['receita_bruta'])
        self.assertTrue(np.isfinite(X).all())
        clientes, perfis = perfilar(rfm, np.arange(10) % 2)
        self.assertEqual(perfis.Clientes.sum(), len(clientes))
        self.assertAlmostEqual(perfis.PercentualReceita.sum(), 100)

    def test_esquema_invalido(self):
        with self.assertRaisesRegex(ValueError, 'Colunas ausentes'):
            preparar(pd.DataFrame({'a':[1]}))

    def test_base_sem_compras(self):
        df = self.base()
        df['Quantity'] = 0
        with self.assertRaisesRegex(ValueError, 'Nenhuma venda válida'):
            preparar(df)


if __name__ == '__main__':
    unittest.main()
