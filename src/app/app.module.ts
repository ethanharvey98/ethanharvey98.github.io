import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';

import { AppRoutingModule } from './app-routing.module';
import { AppComponent } from './app.component';
import { PeopleComponent } from './people/people.component';
import { HciAtTuftsComponent } from './hci-at-tufts/hci-at-tufts.component';
import { CodeAndDatasetsComponent } from './code-and-datasets/code-and-datasets.component';
import { AdmissionsComponent } from './admissions/admissions.component';
import { PublicationsComponent } from './publications/publications.component';
import { ProjectsComponent } from './projects/projects.component';

@NgModule({
  declarations: [
    AppComponent,
    PeopleComponent,
    HciAtTuftsComponent,
    CodeAndDatasetsComponent,
    AdmissionsComponent,
    PublicationsComponent,
    ProjectsComponent
  ],
  imports: [
    BrowserModule,
    AppRoutingModule
  ],
  providers: [],
  bootstrap: [AppComponent]
})
export class AppModule { }
