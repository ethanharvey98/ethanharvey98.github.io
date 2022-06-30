import { Component } from '@angular/core';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css'],
})
export class AppComponent {
  title = 'myApp';

  isHomePage(): boolean {
    return (window.location.href === "http://localhost:4200/" ||
    window.location.href === "https://ethanharvey98.github.io");
  }
}
