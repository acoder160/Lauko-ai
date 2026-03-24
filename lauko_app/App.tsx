/**
 * Lauko AI - Sprint 1.2 & 2.1: UI Polish & Dynamic Location Context
 */

import React, { useState, useRef, useEffect } from 'react';
import Markdown from 'react-native-markdown-display';
import * as DocumentPicker from 'expo-document-picker';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  FlatList,
  StyleSheet,
  KeyboardAvoidingView,
  Platform,
  ActivityIndicator,
  Keyboard
} from 'react-native';
import axios from 'axios';
import { api } from './services/api'; 

interface Message {
  id: string;
  text: string;
  sender: 'user' | 'bot';
}

export default function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputText, setInputText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [selectedFile, setSelectedFile] = useState<DocumentPicker.DocumentPickerAsset | null>(null);
  const [userLocation, setUserLocation] = useState<string | null>(null);
  
  const flatListRef = useRef<FlatList>(null);

  // Fetch user location automatically on app load
  useEffect(() => {
    const fetchLocation = async () => {
      try {
        const response = await axios.get('https://ipapi.co/json/');
        if (response.data && response.data.city && response.data.country_name) {
          setUserLocation(`${response.data.city}, ${response.data.country_name}`);
        }
      } catch (error) {
        console.warn("[Location Error] Could not fetch location automatically", error);
        // Silently fail, it will fall back to "Unknown Location" on the backend
      }
    };
    
    fetchLocation();
  }, []);

  const handlePickDocument = async () => {
    try {
      const result = await DocumentPicker.getDocumentAsync({
        type: ['application/pdf', 'text/plain'],
      });

      if (!result.canceled) {
        setSelectedFile(result.assets[0]);
      }
    } catch (error) {
      console.error("[Picker Error]:", error);
    }
  };

  const removeSelectedFile = () => {
    setSelectedFile(null);
  };

  const handleSend = async () => {
    if (!inputText.trim() && !selectedFile) return;

    const currentText = inputText.trim();
    const currentFile = selectedFile;

    const newMessages: Message[] = [];
    if (currentFile) {
      newMessages.push({ 
        id: Date.now().toString() + '_f', 
        text: `📎 **Document uploaded:** ${currentFile.name}`, 
        sender: 'user' 
      });
    }
    if (currentText) {
      newMessages.push({ 
        id: Date.now().toString() + '_t', 
        text: currentText, 
        sender: 'user' 
      });
    }

    setMessages((prev) => [...prev, ...newMessages]);
    setInputText('');
    setSelectedFile(null);
    setIsLoading(true);
    
    // Dismiss keyboard on send for a cleaner feel, user can tap to type again
    Keyboard.dismiss();

    try {
      if (currentFile) {
        const formData = new FormData();
        if (Platform.OS === 'web') {
          formData.append('file', currentFile.file as Blob);
        } else {
          // @ts-ignore
          formData.append('file', { uri: currentFile.uri, name: currentFile.name, type: currentFile.mimeType });
        }

        // Make sure to use your current network IP for the file upload URL as well
        await axios.post(
          `http://192.168.1.139:8000/api/v1/upload-file?user_id=developer_1`, 
          formData,
          { headers: { 'Content-Type': 'multipart/form-data' } }
        );
      }

      let textToSend = currentText;
      if (currentFile && !currentText) {
        textToSend = `I just uploaded the file "${currentFile.name}". Please confirm receipt and summarize the key facts you extracted for my dossier.`;
      }

      let botResponseText = "";
      if (textToSend) {
        // Pass the dynamic location to the backend
        const response = await api.sendMessage(textToSend, userLocation);
        botResponseText = response.response;
      }

      setMessages((prev) => [...prev, {
        id: (Date.now() + 1).toString(),
        text: botResponseText,
        sender: 'bot',
      }]);

    } catch (error) {
      setMessages((prev) => [...prev, {
        id: (Date.now() + 1).toString(),
        text: '❌ Network error. The backend is not responding.',
        sender: 'bot',
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  const renderMessage = ({ item }: { item: Message }) => {
    const isUser = item.sender === 'user';
    return (
      <View style={[styles.messageWrapper, isUser ? styles.messageWrapperUser : styles.messageWrapperBot]}>
        {!isUser && (
          <LinearGradient colors={['#007AFF', '#00C6FF']} style={styles.botAvatar}>
            <Ionicons name="sparkles" size={14} color="#FFF" />
          </LinearGradient>
        )}
        <View style={[isUser ? styles.userBubble : styles.botBubble]}>
          {isUser ? (
            <Text style={styles.userText}>{item.text}</Text>
          ) : (
            <Markdown style={markdownStyles}>{item.text}</Markdown>
          )}
        </View>
      </View>
    );
  };

  const renderWelcomeScreen = () => (
    <View style={styles.welcomeContainer}>
      <LinearGradient colors={['#007AFF', '#00C6FF']} style={styles.logoContainer}>
        <Ionicons name="planet" size={48} color="#FFF" />
      </LinearGradient>
      <Text style={styles.welcomeTitle}>Hi, I'm Lauko</Text>
      <Text style={styles.welcomeSubtitle}>Your proactive AI companion. How can I help you today?</Text>
    </View>
  );

  return (
    <KeyboardAvoidingView 
      style={styles.container} 
      behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
      keyboardVerticalOffset={Platform.OS === 'ios' ? 0 : -20} // Fix 1: Negative offset for Android to remove the gap
    >
      <View style={styles.header}>
        <TouchableOpacity style={styles.headerIcon}>
           <Ionicons name="menu-outline" size={28} color="#111827" />
        </TouchableOpacity>
        
        <Text style={styles.headerTitle}>Lauko AI</Text>
        
        <TouchableOpacity style={styles.headerIcon}>
           <Ionicons name="earth-outline" size={24} color="#111827" />
        </TouchableOpacity>
      </View>

      {messages.length === 0 ? (
        renderWelcomeScreen()
      ) : (
        <FlatList
          ref={flatListRef}
          data={messages}
          keyExtractor={(item) => item.id}
          renderItem={renderMessage}
          contentContainerStyle={styles.chatContainer}
          onContentSizeChange={() => flatListRef.current?.scrollToEnd({ animated: true })}
          keyboardShouldPersistTaps="handled" 
        />
      )}

      {isLoading && (
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="small" color="#007AFF" />
          <Text style={styles.loadingText}>Lauko is thinking...</Text>
        </View>
      )}

      {/* REWORKED INPUT AREA */}
      <View style={styles.inputOuterContainer}>
        {selectedFile && (
          <View style={styles.attachmentBadge}>
            <Ionicons name="document-text" size={16} color="#007AFF" />
            <Text style={styles.attachmentText} numberOfLines={1}>{selectedFile.name}</Text>
            <TouchableOpacity onPress={removeSelectedFile}>
              <Ionicons name="close-circle" size={20} color="#999" />
            </TouchableOpacity>
          </View>
        )}

        <View style={styles.inputContainer}>
          {/* Bigger, centered attach button with spacing */}
          <TouchableOpacity style={styles.attachButton} onPress={handlePickDocument} disabled={isLoading}>
            <Ionicons name="add" size={34} color="#007AFF" />
          </TouchableOpacity>

          <TextInput
            style={styles.input}
            placeholder="Message Lauko..."
            placeholderTextColor="#999"
            value={inputText}
            onChangeText={(text) => {
              // Fix 2: Prevent leftover newline from Enter key submission
              if (text === '\n') {
                setInputText('');
              } else {
                setInputText(text);
              }
            }}
            multiline={true}
            blurOnSubmit={false}
            onSubmitEditing={handleSend}
            onKeyPress={(e) => {
              if (e.nativeEvent.key === 'Enter') {
                // Prevent default behavior on supported platforms
                if (e.preventDefault) e.preventDefault();
                handleSend();
              }
            }}
            // Fix 3: Removed editable={!isLoading} so user can type while Lauko is thinking
          />

          {/* Centered send button with spacing */}
          <TouchableOpacity onPress={handleSend} disabled={isLoading || (!inputText.trim() && !selectedFile)}>
            <LinearGradient 
              colors={(!inputText.trim() && !selectedFile) ? ['#E5E5EA', '#E5E5EA'] : ['#007AFF', '#00C6FF']} 
              style={styles.sendButton}
            >
              <Ionicons name="arrow-up" size={20} color="#FFF" />
            </LinearGradient>
          </TouchableOpacity>
        </View>
      </View>
    </KeyboardAvoidingView>
  );
}

// --- STYLES ---
const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#F9FAFB' },
  
  header: {
    paddingTop: Platform.OS === 'web' ? 20 : 50,
    paddingBottom: 15,
    paddingHorizontal: 20,
    backgroundColor: '#FFF',
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    borderBottomWidth: 1,
    borderBottomColor: '#F3F4F6',
  },
  headerTitle: { fontSize: 18, fontWeight: '700', color: '#111827' },
  headerIcon: { padding: 5 },
  
  welcomeContainer: { flex: 1, justifyContent: 'center', alignItems: 'center', padding: 20 },
  logoContainer: { width: 80, height: 80, borderRadius: 24, justifyContent: 'center', alignItems: 'center', marginBottom: 20, shadowColor: '#007AFF', shadowOpacity: 0.3, shadowRadius: 15, elevation: 10 },
  welcomeTitle: { fontSize: 24, fontWeight: 'bold', color: '#111827', marginBottom: 10 },
  welcomeSubtitle: { fontSize: 16, color: '#6B7280', textAlign: 'center', paddingHorizontal: 20, lineHeight: 24 },

  chatContainer: { padding: 15, paddingBottom: 20 },
  messageWrapper: { flexDirection: 'row', marginBottom: 20, alignItems: 'flex-start' },
  messageWrapperUser: { justifyContent: 'flex-end' },
  messageWrapperBot: { justifyContent: 'flex-start' },
  
  botAvatar: { width: 28, height: 28, borderRadius: 14, justifyContent: 'center', alignItems: 'center', marginRight: 12, marginTop: 4 },
  
  userBubble: { maxWidth: '75%', padding: 14, borderRadius: 20, backgroundColor: '#007AFF', borderBottomRightRadius: 4 },
  botBubble: { maxWidth: '85%', paddingRight: 10 },
  userText: { color: '#FFF', fontSize: 16, lineHeight: 22 },
  
  loadingContainer: { flexDirection: 'row', alignItems: 'center', padding: 15, paddingLeft: 55 },
  loadingText: { marginLeft: 10, color: '#6B7280', fontSize: 14 },
  
  // FIXED INPUT AREA STYLES
  inputOuterContainer: { 
    backgroundColor: '#FFF', 
    borderTopWidth: 1, 
    borderTopColor: '#F3F4F6', 
    paddingVertical: 12, // More breathing room
    paddingHorizontal: 15, // Side padding for the whole bar
    paddingBottom: Platform.OS === 'ios' ? 25 : 12 
  },
  attachmentBadge: { flexDirection: 'row', alignItems: 'center', backgroundColor: '#F3F4F6', padding: 8, borderRadius: 12, marginBottom: 10, alignSelf: 'flex-start' },
  attachmentText: { marginLeft: 8, marginRight: 15, color: '#374151', fontSize: 14, fontWeight: '500', maxWidth: 200 },
  
  inputContainer: { 
    flexDirection: 'row', 
    alignItems: 'center' // <--- THIS VERTICALLY CENTERS EVERYTHING
  },
  attachButton: { 
    marginRight: 12, // Un-sticks it from the text input
    justifyContent: 'center',
    alignItems: 'center'
  },
  input: { 
    flex: 1, 
    backgroundColor: '#F3F4F6', 
    borderRadius: 22, 
    paddingHorizontal: 18, 
    paddingTop: Platform.OS === 'web' ? 12 : 14, 
    paddingBottom: Platform.OS === 'web' ? 12 : 14, 
    fontSize: 16, 
    maxHeight: 120, 
    minHeight: 45, 
    color: '#111827' 
  },
  sendButton: { 
    width: 42, 
    height: 42, 
    borderRadius: 21, 
    justifyContent: 'center', 
    alignItems: 'center', 
    marginLeft: 12 // Un-sticks it from the text input
  },
});

const markdownStyles = {
  body: { color: '#1F2937', fontSize: 16, lineHeight: 28 }, 
  strong: { fontWeight: '700', color: '#111827' },
  em: { fontStyle: 'italic', color: '#4B5563' },
  paragraph: { marginTop: 0, marginBottom: 16 }, 
  list_item: { marginVertical: 6 },
  hr: { backgroundColor: '#E5E7EB', height: 1, marginVertical: 16 },
};