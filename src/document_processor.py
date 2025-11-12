"""
Document processor for PDFs and CSV files.
Handles extraction and chunking of documents for RAG.
"""

import os
import pandas as pd
from pathlib import Path
from typing import List, Dict
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter


class DocumentProcessor:
    """Process PDF documents and CSV data for RAG."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
    
    def process_pdf(self, pdf_path: str) -> List[Dict[str, str]]:
        """Extract text from a PDF file and split into chunks."""
        print(f"Processing PDF: {pdf_path}")
        chunks = []
        
        try:
            reader = PdfReader(pdf_path)
            full_text = ""
            
            for page_num, page in enumerate(reader.pages, start=1):
                text = page.extract_text()
                if text.strip():
                    full_text += f"\n\n--- Page {page_num} ---\n\n{text}"
            
            # Split into chunks
            text_chunks = self.text_splitter.split_text(full_text)
            
            for i, chunk in enumerate(text_chunks):
                chunks.append({
                    "content": chunk,
                    "source": os.path.basename(pdf_path),
                    "chunk_id": i,
                    "type": "pdf"
                })
            
            print(f"  Extracted {len(chunks)} chunks from {len(reader.pages)} pages")
            
        except Exception as e:
            print(f"  Error processing {pdf_path}: {e}")
        
        return chunks
    
    def process_csv(self, csv_path: str) -> List[Dict[str, str]]:
        """Process CSV price data and create meaningful text chunks."""
        print(f"Processing CSV: {csv_path}")
        chunks = []
        
        try:
            df = pd.read_csv(csv_path, comment='#')
            
            # Create summary statistics
            summary_chunks = self._create_csv_summaries(df)
            chunks.extend(summary_chunks)
            
            # Create time-series chunks for major commodities
            time_series_chunks = self._create_time_series_chunks(df)
            chunks.extend(time_series_chunks)
            
            # Create regional chunks
            regional_chunks = self._create_regional_chunks(df)
            chunks.extend(regional_chunks)
            
            print(f"  Created {len(chunks)} chunks from CSV data")
            
        except Exception as e:
            print(f"  Error processing {csv_path}: {e}")
        
        return chunks
    
    def _create_csv_summaries(self, df: pd.DataFrame) -> List[Dict[str, str]]:
        """Create summary statistics from CSV."""
        chunks = []
        
        # Overall summary
        if 'commodity' in df.columns and 'price' in df.columns:
            summary_text = f"""
FOOD PRICE DATA SUMMARY FOR ETHIOPIA

Total records: {len(df)}
Date range: {df['date'].min() if 'date' in df.columns else 'N/A'} to {df['date'].max() if 'date' in df.columns else 'N/A'}
Unique commodities: {df['commodity'].nunique() if 'commodity' in df.columns else 'N/A'}
Unique markets: {df['market'].nunique() if 'market' in df.columns else 'N/A'}
Unique regions: {df['admin1'].nunique() if 'admin1' in df.columns else 'N/A'}

Average price by commodity:
"""
            if 'commodity' in df.columns and 'price' in df.columns:
                avg_prices = df.groupby('commodity')['price'].mean().sort_values(ascending=False)
                for commodity, avg_price in avg_prices.head(20).items():
                    summary_text += f"  - {commodity}: {avg_price:.2f} ETB\n"
            
            chunks.append({
                "content": summary_text,
                "source": os.path.basename("wfp_food_prices_eth (1).csv"),
                "chunk_id": "summary",
                "type": "csv_summary"
            })
        
        return chunks
    
    def _create_time_series_chunks(self, df: pd.DataFrame, max_commodities: int = 10) -> List[Dict[str, str]]:
        """Create time-series analysis chunks for major commodities."""
        chunks = []
        
        if 'commodity' not in df.columns or 'date' not in df.columns or 'price' not in df.columns:
            return chunks
        
        # Get top commodities by frequency
        top_commodities = df['commodity'].value_counts().head(max_commodities).index
        
        for commodity in top_commodities:
            commodity_df = df[df['commodity'] == commodity].copy()
            
            if len(commodity_df) < 10:  # Skip if too few data points
                continue
            
            # Convert date if needed
            try:
                commodity_df['date'] = pd.to_datetime(commodity_df['date'])
                commodity_df = commodity_df.sort_values('date')
                
                # Get price statistics
                min_price = commodity_df['price'].min()
                max_price = commodity_df['price'].max()
                avg_price = commodity_df['price'].mean()
                recent_price = commodity_df['price'].iloc[-1] if len(commodity_df) > 0 else None
                
                # Create time series description
                time_series_text = f"""
PRICE TIME SERIES DATA: {commodity}

Total price observations: {len(commodity_df)}
Date range: {commodity_df['date'].min()} to {commodity_df['date'].max()}
Price range: {min_price:.2f} - {max_price:.2f} ETB
Average price: {avg_price:.2f} ETB
Most recent price: {recent_price:.2f} ETB

Price trend: The price has {'increased' if recent_price and recent_price > avg_price else 'decreased'} 
compared to the average over the period.

Key markets: {', '.join(commodity_df['market'].value_counts().head(5).index.tolist())}
"""
                
                chunks.append({
                    "content": time_series_text,
                    "source": "wfp_food_prices_eth (1).csv",
                    "chunk_id": f"timeseries_{commodity}",
                    "type": "csv_timeseries"
                })
            except Exception as e:
                continue
        
        return chunks
    
    def _create_regional_chunks(self, df: pd.DataFrame, max_regions: int = 10) -> List[Dict[str, str]]:
        """Create regional analysis chunks."""
        chunks = []
        
        if 'admin1' not in df.columns or 'commodity' not in df.columns or 'price' not in df.columns:
            return chunks
        
        # Get top regions
        top_regions = df['admin1'].value_counts().head(max_regions).index
        
        for region in top_regions:
            region_df = df[df['admin1'] == region].copy()
            
            if len(region_df) < 10:
                continue
            
            regional_text = f"""
REGIONAL FOOD PRICE DATA: {region}

Total records: {len(region_df)}
Unique commodities: {region_df['commodity'].nunique()}
Unique markets: {region_df['market'].nunique()}

Average prices by commodity in {region}:
"""
            avg_prices = region_df.groupby('commodity')['price'].mean().sort_values(ascending=False)
            for commodity, avg_price in avg_prices.head(15).items():
                regional_text += f"  - {commodity}: {avg_price:.2f} ETB\n"
            
            regional_text += f"\nMarkets in {region}: {', '.join(region_df['market'].unique()[:10])}"
            
            chunks.append({
                "content": regional_text,
                "source": "wfp_food_prices_eth (1).csv",
                "chunk_id": f"region_{region}",
                "type": "csv_regional"
            })
        
        return chunks
    
    def process_all_documents(self, directory: str = "documents") -> List[Dict[str, str]]:
        """
        Process all documents in directory.
        Defaults to 'documents/' folder.
        """
        all_chunks = []
        directory_path = Path(directory)
        
        # Process PDFs
        pdf_files = list(directory_path.glob("*.pdf"))
        for pdf_file in pdf_files:
            chunks = self.process_pdf(str(pdf_file))
            all_chunks.extend(chunks)
        
        # Process CSV files
        csv_files = list(directory_path.glob("*.csv"))
        for csv_file in csv_files:
            chunks = self.process_csv(str(csv_file))
            all_chunks.extend(chunks)
        
        print(f"\nTotal chunks created: {len(all_chunks)}")
        return all_chunks

