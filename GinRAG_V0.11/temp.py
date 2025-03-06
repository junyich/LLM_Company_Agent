def decode_garbled_text(garbled_text, from_encoding='gbk', to_encoding='utf-8'):
    try:
        # 先把乱码字符串按错误编码还原成 bytes
        byte_data = garbled_text.encode(from_encoding)
        # 再把 bytes 按正确编码解码成中文
        decoded_text = byte_data.decode(to_encoding)
        return decoded_text
    except Exception as e:
        return f"解码失败: {str(e)}"

garbled_string = "σ£¿Σ╕ìσÉîσ£░τÉåΣ╜ìτ╜«∩╝îσ╜▒σôìσ£░Φí¿σÅìτàºτÄç∩╝êAlbedo∩╝ëτÜäΣ╕╗Φªüσ¢áτ┤áµÿ»Σ╗ÇΣ╣ê∩╝ƒ"
decoded = decode_garbled_text(garbled_string)
print(decoded)
