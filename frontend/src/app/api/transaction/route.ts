import { NextResponse } from 'next/server'
import { loadTransactions } from '../../lib/csvUtils'

export async function GET() {
  const transaction = loadTransactions()
  return NextResponse.json(transaction)
}