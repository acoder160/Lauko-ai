import axios from 'axios';

const API_URL = 'http://192.168.1.149:8000/api/v1';

// Hardcoding the user ID for local development.
// TODO: Replace with dynamic UUID after implementing Auth integration.
const USER_ID = 'developer_1'; 


export const api = {
  async sendMessage(text: string, location: string | null = null) {
    try {
      const response = await axios.post(`${API_URL}/chat`, {
        message: text,
        user_id: USER_ID,
        location: location // Send the dynamic location
      });
      return response.data;
    } catch (error) {
      console.error("[API Error] Failed to send message:", error);
      throw error;
    }
  },

  async getProfile() {
    try {
      const response = await axios.get(`${API_URL}/profile/${USER_ID}`);
      return response.data;
    } catch (error) {
      console.error("[API Error] Failed to fetch user profile:", error);
      throw error;
    }
  }
};