import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

# =========================================================================
# STEP 1: ĐỌC FILE DỮ LIỆU THỰC TẾ (Cấu hình từ ảnh màn hình của bạn)
# =========================================================================
file_path = r"C:\Users\asus\Downloads\PS_20174392719_1491204439457_log.xlsx"

print("--- BẮT ĐẦU QUY TRÌNH TIỀN XỬ LÝ DỮ LIỆU KẾ TOÁN ---")
print(f"Đang đọc dữ liệu từ file: {file_path}...")

try:
    # Đọc file Excel
    df = pd.read_excel(file_path)
    print(f"-> Tải dữ liệu thành công! Kích thước: {df.shape[0]} dòng, {df.shape[1]} cột.")
except Exception as e:
    print(f"-> LỖI: Không thể đọc được file. Hãy kiểm tra lại file đã tắt chưa hoặc đuôi file có phải .csv không.")
    print(f"Chi tiết lỗi: {e}")

# Hiển thị danh sách các cột để bạn dễ quan sát và đối chiếu tên cột
print("\nDanh sách các cột hiện có trong file của bạn:")
print(list(df.columns))


# =========================================================================
# STEP 2: KIỂM TRA VÀ XỬ LÝ DỮ LIỆU KHUYẾT (MISSING VALUES)
# =========================================================================
print("\n[1] Đang kiểm tra ô trống/dữ liệu khuyết...")
missing_summary = df.isnull().sum()
print("Số lượng ô trống ở mỗi cột:")
print(missing_summary[missing_summary > 0] if missing_summary.sum() > 0 else "Không có ô trống nào!")

# Tự động điền dữ liệu khuyết nếu có:
# - Cột dạng Số (Numeric): Điền bằng giá trị 0
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].fillna(0)

# - Cột dạng Chữ/Định tính (Object/Category): Điền bằng chữ "CHUA_RO"
categorical_cols = df.select_dtypes(include=['object']).columns
df[categorical_cols] = df[categorical_cols].fillna("CHUA_RO")


# =========================================================================
# STEP 3: MÃ HÓA BIẾN ĐỊNH TÍNH SANG SỐ (CATEGORICAL ENCODING)
# =========================================================================
print("\n[2] Đang chuyển đổi các biến định tính (Mã tài khoản/Nội dung) sang dạng số...")

# Tạo một bảng dữ liệu mới chỉ chứa các đặc trưng số để chuẩn bị đưa vào AI
df_features = pd.DataFrame()

# Tự động copy các cột số hiện tại sang bảng đặc trưng mới
for col in numeric_cols:
    df_features[col] = df[col]

# Mã hóa toàn bộ các cột chữ sang số bằng LabelEncoder
label_encoders = {}
for col in categorical_cols:
    # Bỏ qua các cột định danh không mang tính phân loại như Mã chứng từ/ID nếu có
    if 'id' in col.lower() or 'ma_ct' in col.lower() or 'chung_tu' in col.lower():
        continue
    
    le = LabelEncoder()
    df_features[f'{col}_Maho'] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le
    print(f"   > Đã mã hóa cột định tính: [{col}]")


# =========================================================================
# STEP 4: CHUẨN HÓA ĐƯA DỮ LIỆU VỀ CÙNG THANG ĐO (Z-SCORE SCALING)
# =========================================================================
print("\n[3] Đang tiến hành chuẩn hóa thang đo dữ liệu (Z-score)...")

if not df_features.empty:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_features)
    
    # Chuyển đổi mảng NumPy sau khi scale ngược lại thành DataFrame để dễ quan sát
    df_scaled_preview = pd.DataFrame(X_scaled, columns=df_features.columns)
    
    print("\n--- TIỀN XỬ LÝ HOÀN TẤT ---")
    print(f"Kích thước ma trận dữ liệu đầu vào cho AI: {X_scaled.shape}")
    print("\nXem trước 5 dòng dữ liệu đã chuẩn hóa hoàn toàn (Sẵn sàng cho mô hình):")
    print(df_scaled_preview.head())
else:
    print("-> LỖI: Không trích xuất được đặc trưng nào hợp lệ để chuẩn hóa.")