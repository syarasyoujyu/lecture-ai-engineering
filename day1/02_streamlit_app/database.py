# database.py
import pandas as pd
from datetime import datetime
import streamlit as st
import os
from supabase import create_client
from metrics import calculate_metrics  # metricsを計算するために必要

# --- Supabase設定 ---
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- テーブル名の定義 ---
TABLE_NAME = "chat_history"


# --- データベース初期化 ---
def init_db():
    """データベースとテーブルを初期化する"""
    try:
        # Supabaseではテーブルは管理画面で作成するため、
        # この関数では接続テストのみを行う
        _ = supabase.table(TABLE_NAME).select("id").limit(1).execute()
        print("Supabase connection test successful.")
    except Exception as e:
        st.error(f"Supabaseの接続テストに失敗しました: {e}")
        raise e


# --- データ操作関数 ---
def save_to_db(question, answer, feedback, correct_answer, is_correct, response_time):
    """チャット履歴と評価指標をデータベースに保存する"""
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 追加の評価指標を計算
        bleu_score, similarity_score, word_count, relevance_score = calculate_metrics(
            answer, correct_answer
        )

        # データを準備
        data = {
            "timestamp": timestamp,
            "question": question,
            "answer": answer,
            "feedback": feedback,
            "correct_answer": correct_answer,
            "is_correct": is_correct,
            "response_time": response_time,
            "bleu_score": bleu_score,
            "similarity_score": similarity_score,
            "word_count": word_count,
            "relevance_score": relevance_score,
        }

        # Supabaseにデータを挿入
        _ = supabase.table(TABLE_NAME).insert(data).execute()
        print("Data saved to Supabase successfully.")
    except Exception as e:
        st.error(f"データベースへの保存中にエラーが発生しました: {e}")


def get_chat_history():
    """データベースから全てのチャット履歴を取得する"""
    try:
        # Supabaseからデータを取得
        response = (
            supabase.table(TABLE_NAME)
            .select("*")
            .order("timestamp", desc=True)
            .execute()
        )

        # DataFrameに変換
        df = pd.DataFrame(response.data)

        # is_correct カラムのデータ型を確認し、必要なら変換
        if "is_correct" in df.columns:
            df["is_correct"] = pd.to_numeric(df["is_correct"], errors="coerce")

        return df
    except Exception as e:
        st.error(f"履歴の取得中にエラーが発生しました: {e}")
        return pd.DataFrame()


def get_db_count():
    """データベース内のレコード数を取得する"""
    try:
        response = supabase.table(TABLE_NAME).select("id").execute()
        return len(response.data)
    except Exception as e:
        st.error(f"レコード数の取得中にエラーが発生しました: {e}")
        return 0


def clear_db():
    """データベースの全レコードを削除する"""
    confirmed = st.session_state.get("confirm_clear", False)

    if not confirmed:
        st.warning(
            "本当にすべてのデータを削除しますか？もう一度「データベースをクリア」ボタンを押すと削除が実行されます。"
        )
        st.session_state.confirm_clear = True
        return False

    try:
        # Supabaseの全レコード削除
        _ = supabase.table(TABLE_NAME).delete().neq("id", 0).execute()
        st.success("データベースが正常にクリアされました。")
        st.session_state.confirm_clear = False
        return True
    except Exception as e:
        st.error(f"データベースのクリア中にエラーが発生しました: {e}")
        st.session_state.confirm_clear = False
        return False
