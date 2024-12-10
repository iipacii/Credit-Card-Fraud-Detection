// src/app/api/svm/route.ts
import { NextRequest, NextResponse } from 'next/server'
import { Transaction } from '@/types'
import dotenv from 'dotenv'

dotenv.config()
const backendUrl = process.env.BACKEND_URL
export async function POST(request: NextRequest) {
  const transaction: Transaction = await request.json()
  console.log('Received transaction in POST:', transaction.transaction_id)

  const response = await fetch(`${backendUrl}/api/naivebayes`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(transaction),
  });

  const result = await response.json();

  return NextResponse.json(result);
}