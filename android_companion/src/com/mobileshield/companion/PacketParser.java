package com.mobileshield.companion;

import java.nio.ByteBuffer;
import java.util.Locale;

public class PacketParser {

    public static class ParsedPacket {
        public double timestamp;
        public String srcIp;
        public String dstIp;
        public int srcPort;
        public int dstPort;
        public int protocol; // 6=TCP, 17=UDP, 1=ICMP
        public int packetLen;
        public int tcpFlags;
        public int ttl;
        public String rawHex;
        public boolean isValid;
    }

    public static ParsedPacket parse(byte[] data, int length) {
        ParsedPacket p = new ParsedPacket();
        p.timestamp = System.currentTimeMillis() / 1000.0;
        p.packetLen = length;
        p.isValid = false;

        if (length < 20) {
            return p;
        }

        int version = (data[0] >> 4) & 0x0F;
        if (version == 4) {
            int ihl = (data[0] & 0x0F) * 4;
            p.ttl = data[8] & 0xFF;
            p.protocol = data[9] & 0xFF;

            p.srcIp = String.format(Locale.US, "%d.%d.%d.%d",
                    data[12] & 0xFF, data[13] & 0xFF, data[14] & 0xFF, data[15] & 0xFF);
            p.dstIp = String.format(Locale.US, "%d.%d.%d.%d",
                    data[16] & 0xFF, data[17] & 0xFF, data[18] & 0xFF, data[19] & 0xFF);

            if (p.protocol == 6 && length >= ihl + 20) { // TCP
                p.srcPort = ((data[ihl] & 0xFF) << 8) | (data[ihl + 1] & 0xFF);
                p.dstPort = ((data[ihl + 2] & 0xFF) << 8) | (data[ihl + 3] & 0xFF);
                p.tcpFlags = data[ihl + 13] & 0x3F;
                p.isValid = true;
            } else if (p.protocol == 17 && length >= ihl + 8) { // UDP
                p.srcPort = ((data[ihl] & 0xFF) << 8) | (data[ihl + 1] & 0xFF);
                p.dstPort = ((data[ihl + 2] & 0xFF) << 8) | (data[ihl + 3] & 0xFF);
                p.isValid = true;
            } else {
                p.srcPort = 0;
                p.dstPort = 0;
                p.isValid = true;
            }
        } else if (version == 6 && length >= 40) {
            p.protocol = data[6] & 0xFF;
            p.ttl = data[7] & 0xFF;
            // Format standard IPv6 addresses
            StringBuilder srcBuilder = new StringBuilder();
            StringBuilder dstBuilder = new StringBuilder();
            for (int i = 8; i < 24; i += 2) {
                if (srcBuilder.length() > 0) srcBuilder.append(":");
                srcBuilder.append(String.format("%x", ((data[i] & 0xFF) << 8) | (data[i + 1] & 0xFF)));
            }
            for (int i = 24; i < 40; i += 2) {
                if (dstBuilder.length() > 0) dstBuilder.append(":");
                dstBuilder.append(String.format("%x", ((data[i] & 0xFF) << 8) | (data[i + 1] & 0xFF)));
            }
            p.srcIp = srcBuilder.toString();
            p.dstIp = dstBuilder.toString();
            if (length >= 48) {
                p.srcPort = ((data[40] & 0xFF) << 8) | (data[41] & 0xFF);
                p.dstPort = ((data[42] & 0xFF) << 8) | (data[43] & 0xFF);
            }
            p.isValid = true;
        }

        // Convert full raw packet bytes to hex for authentic PCAP recording
        int hexBytesLen = Math.min(length, 65535);
        StringBuilder sb = new StringBuilder(hexBytesLen * 2);
        for (int i = 0; i < hexBytesLen; i++) {
            sb.append(String.format("%02x", data[i]));
        }
        p.rawHex = sb.toString();

        return p;
    }
}
