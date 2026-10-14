package com.mobileshield.companion;

import java.net.InetAddress;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Constructs RFC-compliant IPv4, TCP, and UDP packets for TUN return traffic injection.
 */
public class PacketCrafter {

    private static final AtomicInteger ID_GENERATOR = new AtomicInteger(1000);

    public static byte[] ipToBytes(String ipStr) {
        try {
            return InetAddress.getByName(ipStr).getAddress();
        } catch (Exception e) {
            return new byte[]{0, 0, 0, 0};
        }
    }

    public static int computeIpChecksum(byte[] buf, int offset, int length) {
        long sum = 0;
        for (int i = 0; i < length; i += 2) {
            int word = ((buf[offset + i] & 0xFF) << 8) | (buf[offset + i + 1] & 0xFF);
            sum += word;
        }
        while ((sum >> 16) > 0) {
            sum = (sum & 0xFFFF) + (sum >> 16);
        }
        return (int) (~sum & 0xFFFF);
    }

    public static int computeTransportChecksum(byte[] ipHdr, int ipHdrOffset, byte[] transportData, int transportLen, int protocol) {
        long sum = 0;
        // Pseudo-header: Source IP (4 bytes) + Dest IP (4 bytes)
        for (int i = 12; i < 20; i += 2) {
            int word = ((ipHdr[ipHdrOffset + i] & 0xFF) << 8) | (ipHdr[ipHdrOffset + i + 1] & 0xFF);
            sum += word;
        }
        // Zero (1 byte) + Protocol (1 byte)
        sum += protocol & 0xFF;
        // Transport length (2 bytes)
        sum += transportLen & 0xFFFF;

        // Transport header and payload
        for (int i = 0; i < transportLen - 1; i += 2) {
            int word = ((transportData[i] & 0xFF) << 8) | (transportData[i + 1] & 0xFF);
            sum += word;
        }
        if (transportLen % 2 != 0) {
            sum += (transportData[transportLen - 1] & 0xFF) << 8;
        }

        while ((sum >> 16) > 0) {
            sum = (sum & 0xFFFF) + (sum >> 16);
        }
        int res = (int) (~sum & 0xFFFF);
        return res == 0 ? 0xFFFF : res;
    }

    public static byte[] buildUdpPacket(String srcIp, String dstIp, int srcPort, int dstPort, byte[] payload) {
        int payloadLen = payload != null ? payload.length : 0;
        int udpLen = 8 + payloadLen;
        int totalLen = 20 + udpLen;
        byte[] pkt = new byte[totalLen];

        // IPv4 Header
        pkt[0] = 0x45; // Version 4, IHL 5
        pkt[1] = 0x00; // DSCP / ECN
        pkt[2] = (byte) ((totalLen >> 8) & 0xFF);
        pkt[3] = (byte) (totalLen & 0xFF);
        int ident = ID_GENERATOR.incrementAndGet();
        pkt[4] = (byte) ((ident >> 8) & 0xFF);
        pkt[5] = (byte) (ident & 0xFF);
        pkt[6] = 0x40; // Don't fragment
        pkt[7] = 0x00;
        pkt[8] = 64;   // TTL
        pkt[9] = 17;   // Protocol UDP
        pkt[10] = 0;   // Checksum placeholder
        pkt[11] = 0;

        byte[] srcBytes = ipToBytes(srcIp);
        byte[] dstBytes = ipToBytes(dstIp);
        System.arraycopy(srcBytes, 0, pkt, 12, 4);
        System.arraycopy(dstBytes, 0, pkt, 16, 4);

        int ipCksum = computeIpChecksum(pkt, 0, 20);
        pkt[10] = (byte) ((ipCksum >> 8) & 0xFF);
        pkt[11] = (byte) (ipCksum & 0xFF);

        // UDP Header
        pkt[20] = (byte) ((srcPort >> 8) & 0xFF);
        pkt[21] = (byte) (srcPort & 0xFF);
        pkt[22] = (byte) ((dstPort >> 8) & 0xFF);
        pkt[23] = (byte) (dstPort & 0xFF);
        pkt[24] = (byte) ((udpLen >> 8) & 0xFF);
        pkt[25] = (byte) (udpLen & 0xFF);
        pkt[26] = 0; // Checksum placeholder
        pkt[27] = 0;

        if (payloadLen > 0) {
            System.arraycopy(payload, 0, pkt, 28, payloadLen);
        }

        byte[] udpBytes = new byte[udpLen];
        System.arraycopy(pkt, 20, udpBytes, 0, udpLen);
        int udpCksum = computeTransportChecksum(pkt, 0, udpBytes, udpLen, 17);
        pkt[26] = (byte) ((udpCksum >> 8) & 0xFF);
        pkt[27] = (byte) (udpCksum & 0xFF);

        return pkt;
    }

    public static byte[] buildTcpPacket(
            String srcIp, String dstIp, int srcPort, int dstPort,
            long seq, long ack, int flags, int window, byte[] payload
    ) {
        int payloadLen = payload != null ? payload.length : 0;
        boolean isSyn = (flags & 0x02) != 0;
        int tcpHdrLen = isSyn ? 24 : 20; // Include MSS option on SYN
        int tcpTotalLen = tcpHdrLen + payloadLen;
        int totalLen = 20 + tcpTotalLen;
        byte[] pkt = new byte[totalLen];

        // IPv4 Header
        pkt[0] = 0x45;
        pkt[1] = 0x00;
        pkt[2] = (byte) ((totalLen >> 8) & 0xFF);
        pkt[3] = (byte) (totalLen & 0xFF);
        int ident = ID_GENERATOR.incrementAndGet();
        pkt[4] = (byte) ((ident >> 8) & 0xFF);
        pkt[5] = (byte) (ident & 0xFF);
        pkt[6] = 0x40; // DF
        pkt[7] = 0x00;
        pkt[8] = 64;   // TTL
        pkt[9] = 6;    // Protocol TCP
        pkt[10] = 0;
        pkt[11] = 0;

        byte[] srcBytes = ipToBytes(srcIp);
        byte[] dstBytes = ipToBytes(dstIp);
        System.arraycopy(srcBytes, 0, pkt, 12, 4);
        System.arraycopy(dstBytes, 0, pkt, 16, 4);

        int ipCksum = computeIpChecksum(pkt, 0, 20);
        pkt[10] = (byte) ((ipCksum >> 8) & 0xFF);
        pkt[11] = (byte) (ipCksum & 0xFF);

        // TCP Header
        int tcpOffset = 20;
        pkt[tcpOffset] = (byte) ((srcPort >> 8) & 0xFF);
        pkt[tcpOffset + 1] = (byte) (srcPort & 0xFF);
        pkt[tcpOffset + 2] = (byte) ((dstPort >> 8) & 0xFF);
        pkt[tcpOffset + 3] = (byte) (dstPort & 0xFF);

        // Sequence number
        pkt[tcpOffset + 4] = (byte) ((seq >> 24) & 0xFF);
        pkt[tcpOffset + 5] = (byte) ((seq >> 16) & 0xFF);
        pkt[tcpOffset + 6] = (byte) ((seq >> 8) & 0xFF);
        pkt[tcpOffset + 7] = (byte) (seq & 0xFF);

        // Acknowledgment number
        pkt[tcpOffset + 8] = (byte) ((ack >> 24) & 0xFF);
        pkt[tcpOffset + 9] = (byte) ((ack >> 16) & 0xFF);
        pkt[tcpOffset + 10] = (byte) ((ack >> 8) & 0xFF);
        pkt[tcpOffset + 11] = (byte) (ack & 0xFF);

        // Data offset & Flags
        pkt[tcpOffset + 12] = (byte) ((tcpHdrLen / 4) << 4);
        pkt[tcpOffset + 13] = (byte) (flags & 0xFF);

        // Window size
        pkt[tcpOffset + 14] = (byte) ((window >> 8) & 0xFF);
        pkt[tcpOffset + 15] = (byte) (window & 0xFF);

        pkt[tcpOffset + 16] = 0; // Checksum placeholder
        pkt[tcpOffset + 17] = 0;
        pkt[tcpOffset + 18] = 0; // Urgent pointer
        pkt[tcpOffset + 19] = 0;

        if (isSyn) {
            // TCP Option: MSS = 1460 (0x05B4)
            pkt[tcpOffset + 20] = 0x02; // Option Kind: MSS
            pkt[tcpOffset + 21] = 0x04; // Option Length: 4
            pkt[tcpOffset + 22] = 0x05;
            pkt[tcpOffset + 23] = (byte) 0xB4;
        }

        if (payloadLen > 0) {
            System.arraycopy(payload, 0, pkt, tcpOffset + tcpHdrLen, payloadLen);
        }

        byte[] tcpBytes = new byte[tcpTotalLen];
        System.arraycopy(pkt, tcpOffset, tcpBytes, 0, tcpTotalLen);
        int tcpCksum = computeTransportChecksum(pkt, 0, tcpBytes, tcpTotalLen, 6);
        pkt[tcpOffset + 16] = (byte) ((tcpCksum >> 8) & 0xFF);
        pkt[tcpOffset + 17] = (byte) (tcpCksum & 0xFF);

        return pkt;
    }

    public static byte[] buildTcpRst(String srcIp, String dstIp, int srcPort, int dstPort, long seq, long ack) {
        return buildTcpPacket(srcIp, dstIp, srcPort, dstPort, seq, ack, 0x14, 0, null); // RST | ACK
    }

    public static int computeTransportChecksumV6(byte[] srcIpBytes, byte[] dstIpBytes, byte[] transportData, int transportLen, int protocol) {
        long sum = 0;
        // 1. Source IPv6 (16 bytes)
        for (int i = 0; i < 16; i += 2) {
            sum += ((srcIpBytes[i] & 0xFF) << 8) | (srcIpBytes[i + 1] & 0xFF);
        }
        // 2. Dest IPv6 (16 bytes)
        for (int i = 0; i < 16; i += 2) {
            sum += ((dstIpBytes[i] & 0xFF) << 8) | (dstIpBytes[i + 1] & 0xFF);
        }
        // 3. Upper-layer packet length (32-bit in pseudo header)
        sum += (transportLen >> 16) & 0xFFFF;
        sum += transportLen & 0xFFFF;
        // 4. Next header
        sum += protocol & 0xFF;

        // 5. Transport payload
        for (int i = 0; i < transportLen - 1; i += 2) {
            sum += ((transportData[i] & 0xFF) << 8) | (transportData[i + 1] & 0xFF);
        }
        if (transportLen % 2 != 0) {
            sum += (transportData[transportLen - 1] & 0xFF) << 8;
        }

        while ((sum >> 16) > 0) {
            sum = (sum & 0xFFFF) + (sum >> 16);
        }
        int res = (int) (~sum & 0xFFFF);
        return res == 0 ? 0xFFFF : res;
    }

    public static byte[] buildUdpPacketV6(String srcIp, String dstIp, int srcPort, int dstPort, byte[] payload) {
        int payloadLen = payload != null ? payload.length : 0;
        int udpLen = 8 + payloadLen;
        int totalLen = 40 + udpLen;
        byte[] pkt = new byte[totalLen];

        // IPv6 Header (40 bytes)
        pkt[0] = 0x60; // Version 6
        pkt[1] = 0x00;
        pkt[2] = 0x00;
        pkt[3] = 0x00;
        pkt[4] = (byte) ((udpLen >> 8) & 0xFF);
        pkt[5] = (byte) (udpLen & 0xFF);
        pkt[6] = 17;   // Next header: UDP
        pkt[7] = 64;   // Hop limit

        byte[] srcBytes = ipToBytes(srcIp);
        byte[] dstBytes = ipToBytes(dstIp);
        System.arraycopy(srcBytes, 0, pkt, 8, 16);
        System.arraycopy(dstBytes, 0, pkt, 24, 16);

        // UDP Header
        pkt[40] = (byte) ((srcPort >> 8) & 0xFF);
        pkt[41] = (byte) (srcPort & 0xFF);
        pkt[42] = (byte) ((dstPort >> 8) & 0xFF);
        pkt[43] = (byte) (dstPort & 0xFF);
        pkt[44] = (byte) ((udpLen >> 8) & 0xFF);
        pkt[45] = (byte) (udpLen & 0xFF);
        pkt[46] = 0;
        pkt[47] = 0;

        if (payloadLen > 0) {
            System.arraycopy(payload, 0, pkt, 48, payloadLen);
        }

        byte[] udpBytes = new byte[udpLen];
        System.arraycopy(pkt, 40, udpBytes, 0, udpLen);
        int udpCksum = computeTransportChecksumV6(srcBytes, dstBytes, udpBytes, udpLen, 17);
        pkt[46] = (byte) ((udpCksum >> 8) & 0xFF);
        pkt[47] = (byte) (udpCksum & 0xFF);

        return pkt;
    }

    public static byte[] buildTcpPacketV6(
            String srcIp, String dstIp, int srcPort, int dstPort,
            long seq, long ack, int flags, int window, byte[] payload
    ) {
        int payloadLen = payload != null ? payload.length : 0;
        boolean isSyn = (flags & 0x02) != 0;
        int tcpHdrLen = isSyn ? 24 : 20;
        int tcpTotalLen = tcpHdrLen + payloadLen;
        int totalLen = 40 + tcpTotalLen;
        byte[] pkt = new byte[totalLen];

        // IPv6 Header (40 bytes)
        pkt[0] = 0x60;
        pkt[1] = 0x00;
        pkt[2] = 0x00;
        pkt[3] = 0x00;
        pkt[4] = (byte) ((tcpTotalLen >> 8) & 0xFF);
        pkt[5] = (byte) (tcpTotalLen & 0xFF);
        pkt[6] = 6;    // Next header: TCP
        pkt[7] = 64;   // Hop limit

        byte[] srcBytes = ipToBytes(srcIp);
        byte[] dstBytes = ipToBytes(dstIp);
        System.arraycopy(srcBytes, 0, pkt, 8, 16);
        System.arraycopy(dstBytes, 0, pkt, 24, 16);

        int tcpOffset = 40;
        pkt[tcpOffset] = (byte) ((srcPort >> 8) & 0xFF);
        pkt[tcpOffset + 1] = (byte) (srcPort & 0xFF);
        pkt[tcpOffset + 2] = (byte) ((dstPort >> 8) & 0xFF);
        pkt[tcpOffset + 3] = (byte) (dstPort & 0xFF);

        pkt[tcpOffset + 4] = (byte) ((seq >> 24) & 0xFF);
        pkt[tcpOffset + 5] = (byte) ((seq >> 16) & 0xFF);
        pkt[tcpOffset + 6] = (byte) ((seq >> 8) & 0xFF);
        pkt[tcpOffset + 7] = (byte) (seq & 0xFF);

        pkt[tcpOffset + 8] = (byte) ((ack >> 24) & 0xFF);
        pkt[tcpOffset + 9] = (byte) ((ack >> 16) & 0xFF);
        pkt[tcpOffset + 10] = (byte) ((ack >> 8) & 0xFF);
        pkt[tcpOffset + 11] = (byte) (ack & 0xFF);

        pkt[tcpOffset + 12] = (byte) ((tcpHdrLen / 4) << 4);
        pkt[tcpOffset + 13] = (byte) (flags & 0xFF);

        pkt[tcpOffset + 14] = (byte) ((window >> 8) & 0xFF);
        pkt[tcpOffset + 15] = (byte) (window & 0xFF);

        pkt[tcpOffset + 16] = 0;
        pkt[tcpOffset + 17] = 0;
        pkt[tcpOffset + 18] = 0;
        pkt[tcpOffset + 19] = 0;

        if (isSyn) {
            pkt[tcpOffset + 20] = 0x02; // MSS
            pkt[tcpOffset + 21] = 0x04;
            pkt[tcpOffset + 22] = 0x05;
            pkt[tcpOffset + 23] = (byte) 0xB4;
        }

        if (payloadLen > 0) {
            System.arraycopy(payload, 0, pkt, tcpOffset + tcpHdrLen, payloadLen);
        }

        byte[] tcpBytes = new byte[tcpTotalLen];
        System.arraycopy(pkt, tcpOffset, tcpBytes, 0, tcpTotalLen);
        int tcpCksum = computeTransportChecksumV6(srcBytes, dstBytes, tcpBytes, tcpTotalLen, 6);
        pkt[tcpOffset + 16] = (byte) ((tcpCksum >> 8) & 0xFF);
        pkt[tcpOffset + 17] = (byte) (tcpCksum & 0xFF);

        return pkt;
    }

    public static byte[] buildTcpRstV6(String srcIp, String dstIp, int srcPort, int dstPort, long seq, long ack) {
        return buildTcpPacketV6(srcIp, dstIp, srcPort, dstPort, seq, ack, 0x14, 0, null);
    }
}
