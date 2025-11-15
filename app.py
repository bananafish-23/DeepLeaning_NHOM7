# Tên tệp: app.py

import streamlit as st
import pandas as pd
import requests
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
import os
import base64
import email
import re
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# --- 1. CẤU HÌNH CƠ BẢN ---
st.set_page_config(page_title="Trợ lý Email AI", layout="wide")


CLIENT_SECRETS_FILE = 'client_secret.json'
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly', 'https://www.googleapis.com/auth/gmail.send']
REDIRECT_URI = 'http://localhost:8501'
BACKEND_URL = "http://127.0.0.1:8000/predict"
DEPARTMENT_MAP = {
    'Incident': 'huypng22416c@st.uel.edu.vn',
    'Request': 'huypng22416c@st.uel.edu.vn',
    'Problem': 'huypng22416c@st.uel.edu.vn',
    'Change': 'huypng22416c@st.uel.edu.vn'
}


# --- HÀM LÀM SẠCH HTML ---
def strip_html_tags(html_content):
    if not html_content: return ""
    cleantext = re.sub(r'<(p|div|tr|li|br)[^>]*>', '\n', html_content, flags=re.IGNORECASE)
    cleantext = re.sub(r'<.*?>', ' ', cleantext)
    cleantext = re.sub(r'&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});', ' ', cleantext)
    cleantext = re.sub(r'[ \t]+', ' ', cleantext)
    cleantext = re.sub(r'\n\s*\n+', '\n\n', cleantext)
    return cleantext.strip()


# --- 2. HÀM XỬ LÝ GOOGLE OAUTH & GMAIL API ---
def get_google_auth_flow():
    return Flow.from_client_secrets_file(CLIENT_SECRETS_FILE, scopes=SCOPES, redirect_uri=REDIRECT_URI)


def get_credentials(auth_code):
    try:
        flow = get_google_auth_flow()
        flow.fetch_token(code=auth_code)
        return flow.credentials
    except Exception as e:
        st.error(f"Lỗi khi lấy token: {e}")
        return None


def get_gmail_service(credentials):
    return build('gmail', 'v1', credentials=credentials)


def parse_email_body(payload):
    try:
        if 'parts' in payload:
            for part in payload['parts']:
                if part['mimeType'] == 'text/plain' and 'data' in part['body']:
                    data = part['body']['data']
                    return base64.urlsafe_b64decode(data).decode('utf-8')
            for part in payload['parts']:
                if part['mimeType'] == 'text/html' and 'data' in part['body']:
                    data = part['body']['data']
                    html_content = base64.urlsafe_b64decode(data).decode('utf-8')
                    return strip_html_tags(html_content)
        elif 'data' in payload['body']:
            data = payload['body']['data']
            content = base64.urlsafe_b64decode(data).decode('utf-8')
            if payload['mimeType'] == 'text/html' or '<html' in content.lower():
                return strip_html_tags(content)
            else:
                return content
        return ""
    except Exception:
        return ""


def fetch_new_emails(service, max_results=10):
    try:
        results = service.users().messages().list(userId='me', labelIds=['INBOX', 'UNREAD'],
                                                  maxResults=max_results).execute()
        messages = results.get('messages', [])
        if not messages: return []
        email_list = []
        for i, msg in enumerate(messages):
            msg_data = service.users().messages().get(userId='me', id=msg['id'], format='full').execute()
            headers = msg_data['payload']['headers']
            email_info = {'index': i + 1, 'id': msg_data['id'], 'snippet': msg_data['snippet'],
                          'subject': 'Không có chủ đề', 'from': 'Không rõ'}
            original_from_val = ""
            original_subject_val = ""
            for header in headers:
                if header['name'] == 'Subject':
                    email_info['subject'] = header['value']
                    original_subject_val = header['value']
                if header['name'] == 'From':
                    email_info['from'] = header['value']
                    original_from_val = header['value']
            body = parse_email_body(msg_data['payload'])
            email_info['body'] = body if body else email_info['snippet']
            email_info['original_from'] = original_from_val
            email_info['original_subject'] = original_subject_val
            email_list.append(email_info)
        return email_list
    except Exception as e:
        st.error(f"Lỗi khi tải email: {e}")
        return []


def forward_email(service, email_data, to_email):
    try:
        new_message = MIMEMultipart()
        new_message['to'] = to_email
        new_message['from'] = "me"
        new_message['subject'] = f"Fwd: {email_data['original_subject']}"
        forward_intro = f"---------- Forwarded message ---------\nFrom: {email_data['original_from']}\nSubject: {email_data['original_subject']}\n\n"
        new_body = forward_intro + email_data['body']
        new_message.attach(MIMEText(new_body, 'plain'))
        raw_message_bytes = new_message.as_bytes()
        raw_message_b64 = base64.urlsafe_b64encode(raw_message_bytes).decode('utf-8')
        body_to_send = {'raw': raw_message_b64}
        service.users().messages().send(
            userId='me',
            body=body_to_send
        ).execute()
        return True
    except KeyError as e:
        st.error(f"Lỗi KeyError khi chuyển tiếp: Thiếu key {e}.")
        return False
    except Exception as e:
        st.error(f"Lỗi API khi gửi email {email_data['id']}: {e}")
        return False


# --- 3. HÀM GỌI API BACKEND ---
def call_prediction_api(email_text):
    try:
        payload = {"text": email_text}
        response = requests.post(BACKEND_URL, json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Lỗi API: {response.status_code} {response.text}"}
    except requests.exceptions.ConnectionError:
        return {"error": "Không thể kết nối đến backend. Bạn đã chạy 'python -m uvicorn main:app' chưa?"}
    except Exception as e:
        return {"error": f"Lỗi không xác định: {e}"}


# --- 4. HÀM NLU  ---
INTENT_KEYWORDS = {
    "LOAD_EMAILS": ["tải", "check", "email", "mail", "hòm thư", "tải lại"],
    "ANALYZE": ["phân tích", "analyse", "xem", "coi", "số", "check mail số"],
    "CONFIRM": ["đồng ý", "ok", "chuyển tiếp", "forward", "yes", "ừ"],
    "CANCEL": ["hủy", "không", "stop", "dừng", "no", "thôi"],
    "CLEAR_CHAT": ["xóa", "clear", "làm mới", "reset", "xóa chat"]
}


def get_intent(prompt):
    prompt_lower = prompt.lower()
    for intent, keywords in INTENT_KEYWORDS.items():
        if any(keyword in prompt_lower for keyword in keywords):
            return intent
    return "UNKNOWN"


def extract_entities(prompt):
    numbers = re.findall(r'\d+', prompt)
    if not numbers: return None
    if len(numbers) == 1: return {"start": int(numbers[0]), "end": int(numbers[0])}
    if len(numbers) >= 2:
        start = int(numbers[0]);
        end = int(numbers[-1])
        if start > end: start, end = end, start
        return {"start": start, "end": end}


# --- 5. HÀM HELPER  ---
def analyze_email(email_index, service, df_emails):
    try:
        email_row = df_emails[df_emails['index'] == email_index].iloc[0]
        email_text = email_row['subject'] + " " + email_row['body']
        api_result = call_prediction_api(email_text)

        if "error" in api_result:
            return f"Lỗi API: {api_result['error']}", None
        else:
            pred_type = api_result['predicted_type']
            pred_pri = api_result['predicted_priority']
            forward_to = DEPARTMENT_MAP.get(pred_type, "Không chuyển tiếp")
            response_content = (
                f"**Phân tích email {email_index} (từ {email_row['from']}):**\n"
                f"* **Loại (Type):** `{pred_type}`\n"
                f"* **Độ ưu tiên (Priority):** `{pred_pri}`\n"
                f"--- \n"
                f"Tôi đề xuất chuyển tiếp đến: **{forward_to}**."
            )
            pending_action = None
            if forward_to != "Không chuyển tiếp":
                pending_action = {
                    'email_data': email_row.to_dict(),
                    'to': forward_to
                }
            return response_content, pending_action
    except IndexError:
        return f"Không tìm thấy email số {email_index} trong danh sách đã tải.", None
    except Exception as e:
        return f"Lỗi khi xử lý email {email_index}: {e}", None


# --- 6. KHỞI TẠO SESSION STATE  ---
if 'credentials' not in st.session_state: st.session_state.credentials = None
if 'emails_df' not in st.session_state: st.session_state.emails_df = pd.DataFrame()
if 'messages' not in st.session_state: st.session_state.messages = []
if 'pending_forward' not in st.session_state: st.session_state.pending_forward = []

# --- 7. GIAO DIỆN STREAMLIT  ---
st.title("🤖 Trợ lý Phân loại Email")

auth_code = st.query_params.get("code")
if st.session_state.credentials is None:
    if auth_code:
        with st.spinner("Đang xác thực..."):
            creds = get_credentials(auth_code)
            if creds:
                st.session_state.credentials = creds
                st.query_params.clear();
                st.rerun()
    else:
        st.warning("Vui lòng đăng nhập bằng tài khoản Google của bạn để tiếp tục.")
        try:
            auth_url, _ = get_google_auth_flow().authorization_url(prompt='consent')
            st.link_button("Đăng nhập với Google", auth_url, use_container_width=True)
        except FileNotFoundError:
            st.error("Lỗi nghiêm trọng: Không tìm thấy tệp `client_secret.json`.");
            st.stop()
        except Exception as e:
            st.error(f"Lỗi khi tạo link đăng nhập: {e}");
            st.stop()
else:
    service = get_gmail_service(st.session_state.credentials)

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if isinstance(message["content"], pd.DataFrame):
                st.dataframe(message["content"], use_container_width=True)
            else:
                st.markdown(message["content"])

    if prompt := st.chat_input("Bạn muốn làm gì? (ví dụ: 'tải email', 'phân tích 5-7', 'xóa')"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        intent = get_intent(prompt)
        entities = extract_entities(prompt)

        if intent == "CLEAR_CHAT":
            st.session_state.messages = []
            st.session_state.emails_df = pd.DataFrame()
            st.session_state.pending_forward = []
            st.rerun()

        else:
            with st.chat_message("assistant"):
                with st.spinner("Bot đang suy nghĩ..."):

                    response_items = []

                    if intent == "CANCEL":
                        st.session_state.pending_forward = []
                        response_items = ["Đã hủy. Bạn muốn làm gì tiếp theo?"]

                    elif intent == "CONFIRM":
                        pending_list = st.session_state.pending_forward
                        if pending_list:
                            success_count = 0
                            fail_count = 0
                            for details in pending_list:
                                success = forward_email(service, details['email_data'], details['to'])
                                if success:
                                    success_count += 1
                                else:
                                    fail_count += 1
                            response_items = [f"✅ Đã chuyển tiếp thành công {success_count} email."]
                            if fail_count > 0:
                                response_items.append(f"❌ Có {fail_count} email bị lỗi khi chuyển tiếp.")
                            st.session_state.pending_forward = []
                        else:
                            response_items = ["Không có email nào đang chờ chuyển tiếp. Bạn muốn phân tích email nào?"]

                    elif intent == "LOAD_EMAILS":
                        st.session_state.pending_forward = []
                        emails_list = fetch_new_emails(service, max_results=10)
                        if emails_list:
                            st.session_state.emails_df = pd.DataFrame(emails_list)
                            response_items = [
                                f"Tôi tìm thấy {len(emails_list)} email mới. Bạn muốn phân tích email số mấy?",
                                st.session_state.emails_df[['index', 'from', 'subject', 'snippet']]
                            ]
                        else:
                            response_items = ["Bạn không có email mới nào."]

                    elif intent == "ANALYZE":
                        st.session_state.pending_forward = []
                        if not entities:
                            response_items = [
                                "Bạn muốn tôi phân tích email **số mấy**? (ví dụ: 'phân tích 3', 'phân tích 5-7')"]
                        elif st.session_state.emails_df.empty:
                            response_items = ["Bạn chưa tải email. Vui lòng gõ 'tải email' trước."]
                        else:
                            indices_to_analyze = list(range(entities["start"], entities["end"] + 1))
                            all_responses_text = []
                            for index in indices_to_analyze:
                                analysis_response, pending_action = analyze_email(
                                    index, service, st.session_state.emails_df
                                )
                                all_responses_text.append(analysis_response)
                                if pending_action:
                                    st.session_state.pending_forward.append(pending_action)
                            response_items = ["\n\n---\n\n".join(all_responses_text)]

                            num_pending = len(st.session_state.pending_forward)
                            if num_pending == 1:
                                response_items.append("Bạn có muốn **đồng ý** chuyển tiếp 1 email này không?")
                            elif num_pending > 1:
                                response_items.append(
                                    f"Bạn có muốn **đồng ý** chuyển tiếp **tất cả {num_pending} email** này không?")
                            else:
                                response_items.append("Phân tích hoàn tất. Không có email nào cần chuyển tiếp.")

                    else:  # intent == "UNKNOWN"
                        response_items = [(
                            "Xin lỗi, tôi không hiểu yêu cầu của bạn. \n\n"
                            "Hãy thử các lệnh sau:\n"
                            "* **'tải email'**: Để kiểm tra 10 email mới nhất.\n"
                            "* **'phân tích 5'**: Để phân tích email số 5.\n"
                            "* **'phân tích 5-8'**: Để phân tích các email từ 5 đến 8.\n"
                            "* **'đồng ý' / 'hủy'**: Để xác nhận hoặc hủy hành động.\n"
                            "* **'xóa'**: Để làm mới cuộc trò chuyện."
                        )]

                for item in response_items:
                    if isinstance(item, pd.DataFrame):
                        st.dataframe(item, use_container_width=True)
                    else:
                        st.markdown(item)
                    st.session_state.messages.append({"role": "assistant", "content": item})