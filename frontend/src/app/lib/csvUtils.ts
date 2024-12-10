import { Transaction } from '@/types'
import fs from 'fs'
import path from 'path'
import { parse } from 'csv-parse/sync'

let transactions: Transaction[] = []
let currentIndex = 0

export const loadTransactions = () => {
  if (transactions.length === 0) {
    const filePath = path.join(process.cwd(), 'public', 'transactions.csv')
    const fileContent = fs.readFileSync(filePath, 'utf-8')
    transactions = parse(fileContent, {
      columns: true,
      skip_empty_lines: true
    })
    transactions = transactions.sort(() => Math.random() - 0.5)
  }

  const transaction = transactions[currentIndex]
  currentIndex = (currentIndex + 1) % transactions.length
  return transaction
}