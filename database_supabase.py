# database_supabase.py
"""
Module untuk manajemen database menggunakan Supabase
"""

import hashlib
import pandas as pd
from typing import Dict, Any, Optional
from datetime import datetime
from supabase import create_client, Client
import streamlit as st


class HistoryDatabase:
    """Manage checking history using Supabase"""
    
    def __init__(self, supabase_url: str = None, supabase_key: str = None):
        """
        Initialize Supabase client
        
        Args:
            supabase_url: Supabase project URL
            supabase_key: Supabase anon/public key
        """
        # Get credentials from parameters or environment/secrets
        self.supabase_url = supabase_url or st.secrets.get("SUPABASE_URL", "")
        self.supabase_key = supabase_key or st.secrets.get("SUPABASE_KEY", "")
        
        if not self.supabase_url or not self.supabase_key:
            st.error("Supabase credentials not found! Please set them in .streamlit/secrets.toml")
            st.stop()
        
        # Initialize Supabase client
        self.supabase: Client = create_client(self.supabase_url, self.supabase_key)
        self.table_name = "news_history"
        
        # Create table if not exists (run this query in Supabase SQL editor)
        self._ensure_table_exists()
    
    def _ensure_table_exists(self):
        """
        Ensure the news_history table exists in Supabase
        Run this SQL in Supabase SQL editor if table doesn't exist:
        """
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS news_history (
            id SERIAL PRIMARY KEY,
            url TEXT NOT NULL,
            url_hash TEXT UNIQUE NOT NULL,
            domain TEXT,
            title TEXT,
            content TEXT,
            prediction TEXT,
            confidence REAL,
            fake_probability REAL,
            real_probability REAL,
            checked_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );

        -- Create index for faster queries
        CREATE INDEX IF NOT EXISTS idx_url_hash ON news_history(url_hash);
        CREATE INDEX IF NOT EXISTS idx_checked_at ON news_history(checked_at DESC);
        CREATE INDEX IF NOT EXISTS idx_domain ON news_history(domain);
        CREATE INDEX IF NOT EXISTS idx_prediction ON news_history(prediction);
        """
        # Note: User needs to run this SQL manually in Supabase dashboard
        return create_table_sql
    
    def add_record(self, record: Dict[str, Any]) -> bool:
        """Add checking record to history"""
        try:
            url_hash = hashlib.md5(record['url'].encode()).hexdigest()
            
            # Prepare data for insertion
            data = {
                'url': record['url'],
                'url_hash': url_hash,
                'domain': record.get('domain', ''),
                'title': record.get('title', ''),
                'content': record.get('content', ''),
                'prediction': record['prediction'],
                'confidence': record['confidence'],
                'fake_probability': record['fake_probability'],
                'real_probability': record['real_probability'],
                'checked_at': record['checked_at'].isoformat() if isinstance(record['checked_at'], datetime) else record['checked_at']
            }
            
            # Try to insert
            response = self.supabase.table(self.table_name).insert(data).execute()
            
            if response.data:
                return True
            
        except Exception as e:
            # If duplicate key error (url_hash already exists), update instead
            if "duplicate key" in str(e).lower():
                return self._update_record(url_hash, record)
            else:
                st.error(f"Error adding record: {str(e)}")
                return False
        
        return False
    
    def _update_record(self, url_hash: str, record: Dict[str, Any]) -> bool:
        """Update existing record"""
        try:
            update_data = {
                'prediction': record['prediction'],
                'confidence': record['confidence'],
                'fake_probability': record['fake_probability'],
                'real_probability': record['real_probability'],
                'checked_at': record['checked_at'].isoformat() if isinstance(record['checked_at'], datetime) else record['checked_at']
            }
            
            response = self.supabase.table(self.table_name)\
                .update(update_data)\
                .eq('url_hash', url_hash)\
                .execute()
            
            return bool(response.data)
            
        except Exception as e:
            st.error(f"Error updating record: {str(e)}")
            return False
    
    def get_history(self, limit: int = 100) -> pd.DataFrame:
        """Get checking history"""
        try:
            # Query with ordering and limit
            response = self.supabase.table(self.table_name)\
                .select("url, domain, title, prediction, confidence, fake_probability, real_probability, checked_at")\
                .order('checked_at', desc=True)\
                .limit(limit)\
                .execute()
            
            if response.data:
                df = pd.DataFrame(response.data)
                # Convert checked_at to datetime if not empty
                if not df.empty and 'checked_at' in df.columns:
                    df['checked_at'] = pd.to_datetime(df['checked_at'])
                return df
            else:
                return pd.DataFrame()
                
        except Exception as e:
            st.error(f"Error fetching history: {str(e)}")
            return pd.DataFrame()
    
    def check_url_exists(self, url: str) -> Optional[Dict[str, Any]]:
        """Check if URL has been checked before"""
        try:
            url_hash = hashlib.md5(url.encode()).hexdigest()
            
            response = self.supabase.table(self.table_name)\
                .select("*")\
                .eq('url_hash', url_hash)\
                .limit(1)\
                .execute()
            
            if response.data and len(response.data) > 0:
                return response.data[0]
            
            return None
            
        except Exception as e:
            st.error(f"Error checking URL: {str(e)}")
            return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            # Total count
            total_response = self.supabase.table(self.table_name)\
                .select("*", count='exact')\
                .execute()
            
            total_count = total_response.count if total_response else 0
            
            # Fake count
            fake_response = self.supabase.table(self.table_name)\
                .select("*", count='exact')\
                .eq('prediction', 'FAKE')\
                .execute()
            
            fake_count = fake_response.count if fake_response else 0
            
            # Get average confidence
            all_data = self.supabase.table(self.table_name)\
                .select("confidence")\
                .execute()
            
            avg_confidence = 0
            if all_data.data:
                confidences = [row['confidence'] for row in all_data.data if row['confidence']]
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            # Get unique domains count
            domains_data = self.supabase.table(self.table_name)\
                .select("domain")\
                .execute()
            
            unique_domains = 0
            if domains_data.data:
                unique_domains = len(set(row['domain'] for row in domains_data.data if row['domain']))
            
            return {
                'total_checks': total_count,
                'fake_count': fake_count,
                'real_count': total_count - fake_count,
                'fake_percentage': (fake_count / total_count * 100) if total_count > 0 else 0,
                'avg_confidence': avg_confidence,
                'unique_domains': unique_domains
            }
            
        except Exception as e:
            st.error(f"Error getting statistics: {str(e)}")
            return {
                'total_checks': 0,
                'fake_count': 0,
                'real_count': 0,
                'fake_percentage': 0,
                'avg_confidence': 0,
                'unique_domains': 0
            }
    
    def search_history(self, search_term: str = "", prediction_filter: str = "All", 
                      domain_filter: str = "", limit: int = 100) -> pd.DataFrame:
        """Search history with filters"""
        try:
            query = self.supabase.table(self.table_name)\
                .select("url, domain, title, prediction, confidence, fake_probability, real_probability, checked_at")\
                .order('checked_at', desc=True)\
                .limit(limit)
            
            # Apply filters
            if prediction_filter != "All":
                query = query.eq('prediction', prediction_filter)
            
            if domain_filter:
                query = query.eq('domain', domain_filter)
            
            if search_term:
                # Supabase text search
                query = query.or_(f"title.ilike.%{search_term}%,url.ilike.%{search_term}%")
            
            response = query.execute()
            
            if response.data:
                df = pd.DataFrame(response.data)
                if not df.empty and 'checked_at' in df.columns:
                    df['checked_at'] = pd.to_datetime(df['checked_at'])
                return df
            else:
                return pd.DataFrame()
                
        except Exception as e:
            st.error(f"Error searching history: {str(e)}")
            return pd.DataFrame()
    
    def get_domain_statistics(self, limit: int = 10) -> pd.DataFrame:
        """Get statistics by domain"""
        try:
            # Get all data for aggregation
            response = self.supabase.table(self.table_name)\
                .select("domain, prediction")\
                .execute()
            
            if response.data:
                df = pd.DataFrame(response.data)
                
                # Group by domain and calculate statistics
                domain_stats = df.groupby('domain').agg({
                    'prediction': [
                        'count',
                        lambda x: (x == 'FAKE').sum(),
                        lambda x: (x == 'REAL').sum(),
                        lambda x: ((x == 'FAKE').sum() / len(x) * 100) if len(x) > 0 else 0
                    ]
                }).round(2)
                
                domain_stats.columns = ['total_checks', 'fake_count', 'real_count', 'fake_rate']
                domain_stats = domain_stats.sort_values('total_checks', ascending=False).head(limit)
                
                return domain_stats.reset_index()
            else:
                return pd.DataFrame()
                
        except Exception as e:
            st.error(f"Error getting domain statistics: {str(e)}")
            return pd.DataFrame()
    
    def delete_record(self, url: str) -> bool:
        """Delete a record by URL"""
        try:
            url_hash = hashlib.md5(url.encode()).hexdigest()
            
            response = self.supabase.table(self.table_name)\
                .delete()\
                .eq('url_hash', url_hash)\
                .execute()
            
            return bool(response.data)
            
        except Exception as e:
            st.error(f"Error deleting record: {str(e)}")
            return False
    
    def export_all_data(self) -> pd.DataFrame:
        """Export all data from database"""
        try:
            response = self.supabase.table(self.table_name)\
                .select("*")\
                .order('checked_at', desc=True)\
                .execute()
            
            if response.data:
                df = pd.DataFrame(response.data)
                if not df.empty and 'checked_at' in df.columns:
                    df['checked_at'] = pd.to_datetime(df['checked_at'])
                return df
            else:
                return pd.DataFrame()
                
        except Exception as e:
            st.error(f"Error exporting data: {str(e)}")
            return pd.DataFrame()